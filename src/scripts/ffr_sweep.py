#!/usr/bin/env python
"""
scripts/ffr_sweep.py
====================
Подбор кальциевых параметров для получения позитивного FFR.

Sweep: Vmax_up × K_NaCa × frequency.
Для каждой комбинации (Vmax_up, K_NaCa) вычисляются биомаркеры на
4 частотах и оценивается качество FFR-кривой по композитной функции потерь.

Запуск:
    python scripts/ffr_sweep.py                 # 9 наборов × 4 частоты, n_jobs=1
    python scripts/ffr_sweep.py --jobs 4        # параллельно на 4 ядрах
    python scripts/ffr_sweep.py --warmup-cycles 200  # быстрее

Целевые признаки позитивного FFR (для здорового человека):
    1. F_systolic растёт с частотой
    2. F_diastolic остаётся низким (< 30% от F_systolic) на всех частотах
    3. F_amplitude монотонно растёт с частотой
    4. Ca_diastolic < 200 нМ даже на 3 Гц
    5. APD90 монотонно убывает с частотой
    6. Ca_nSR_mean растёт с частотой (запас SR увеличивается)

Loss (минимизируем):
    L = w1 * non_monotonic(F_amp)
      + w2 * max(F_dia / F_sys, 0.3) ** 2     — штраф за диастолический тонус
      + w3 * max(Ca_dia[3Hz] - 200, 0)         — штраф за Ca-перегрузку
      + w4 * non_monotonic(APD90, decreasing=True)
      - w5 * (F_amp[3Hz] - F_amp[0.5Hz])       — поощрение роста силы

Результат:
    data/ffr_sweep/
        runs.h5                — все прогоны (4 × 9 = 36)
        biomarkers_all.csv     — биомаркеры для всех комбинаций
        ranking.csv            — комбинации, отсортированные по loss
        sweep_F_systolic.png   — сетка графиков
        sweep_F_amplitude.png
        sweep_Ca_diastolic.png
        sweep_APD90.png
        best_overlay.png       — лучшие 3 комбинации поверх baseline
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# sys.path: корень и src/
_ROOT = Path(__file__).resolve().parent.parent
for d in (_ROOT, _ROOT / "src"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from experiment import SimulationResult, run_single
from model import make_init_state
from parameters import DEFAULT_PARAMS, SimConfig, TNNPMParams
from simulation_io import save_batch

# Импорт биомаркеров из ffr_diagnostic
sys.path.insert(0, str(_ROOT / "scripts"))
from ffr_diagnostic import extract_biomarkers, _last_cycle_indices

# ---------------------------------------------------------------------------
# Sweep grid
# ---------------------------------------------------------------------------

DEFAULT_VMAX_UP_VALUES = [5.8e-4, 8.7e-4, 1.16e-3]  # × 1.0, 1.5, 2.0
DEFAULT_K_NACA_VALUES = [4000.0, 5000.0, 6000.0]  # ± 20%
DEFAULT_FREQUENCIES = [0.5, 1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# Запуск одной точки sweep
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepPoint:
    """Одна точка sweep: (Vmax_up, K_NaCa, frequency)."""

    Vmax_up: float
    K_NaCa: float
    frequency_Hz: float


def run_sweep_point(
    pt: SweepPoint,
    n_cycles_warmup: int,
    n_cycles_record: int,
    atol: float,
    rtol: float,
    max_step: float,
) -> SimulationResult:
    """Один прогон с заданными (Vmax_up, K_NaCa, freq)."""
    period = 1000.0 / pt.frequency_Hz
    p = DEFAULT_PARAMS.replace(
        stim_period=period,
        Vmax_up=pt.Vmax_up,
        K_NaCa=pt.K_NaCa,
    )

    total_ms = (n_cycles_warmup + n_cycles_record) * period
    pts_per_cycle = 2000
    n_out = int(pts_per_cycle * (n_cycles_warmup + n_cycles_record))

    config = SimConfig(
        time_start=0.0,
        time_stop=total_ms,
        atol=atol,
        rtol=rtol,
        max_step=max_step,
        n_out=n_out,
    )

    state0, l0 = make_init_state(p)
    return run_single(p, state0, config, l0=l0)


# ---------------------------------------------------------------------------
# Loss-функция
# ---------------------------------------------------------------------------


def _is_monotonic_increasing(arr: np.ndarray, tol: float = 0.0) -> float:
    """Доля пар (i, i+1), где arr[i+1] >= arr[i] - tol. 1.0 = строго монотонна."""
    if len(arr) < 2:
        return 1.0
    diffs = np.diff(arr)
    return float((diffs >= -tol).sum()) / len(diffs)


def _is_monotonic_decreasing(arr: np.ndarray, tol: float = 0.0) -> float:
    if len(arr) < 2:
        return 1.0
    diffs = np.diff(arr)
    return float((diffs <= tol).sum()) / len(diffs)


def compute_loss(biomarkers_per_freq: list[dict]) -> dict:
    """Композитная функция потерь для одного набора параметров (один Vmax×K_NaCa).

    Аргумент: список биомаркеров по частотам, отсортированный по freq возрастающе.
    """
    freqs = np.array([b["frequency_Hz"] for b in biomarkers_per_freq])
    F_sys = np.array([b["F_systolic_mN"] for b in biomarkers_per_freq])
    F_dia = np.array([b["F_diastolic_mN"] for b in biomarkers_per_freq])
    F_amp = np.array([b["F_amplitude_mN"] for b in biomarkers_per_freq])
    Ca_dia = np.array([b["Ca_diastolic_nM"] for b in biomarkers_per_freq])
    APD = np.array([b["APD90_ms"] for b in biomarkers_per_freq])
    SR = np.array([b["Ca_nSR_mean_mM"] for b in biomarkers_per_freq])

    # ── Компоненты лосса ────────────────────────────────────────────────────

    # 1. Монотонность F_amplitude (хотим растущая)
    mono_F = _is_monotonic_increasing(F_amp)
    loss_mono_F = (1.0 - mono_F) * 100.0

    # 2. Диастолический тонус: штрафуем F_dia/F_sys > 0.30
    ratio_dia = F_dia / np.maximum(F_sys, 1e-3)
    excess_tonus = np.maximum(ratio_dia - 0.30, 0.0)
    loss_tonus = float((excess_tonus**2).sum() * 100.0)

    # 3. Ca-перегрузка на максимальной частоте
    ca_dia_max = float(Ca_dia.max())
    loss_ca = max(ca_dia_max - 200.0, 0.0) * 0.5

    # 4. Монотонность APD (хотим убывающая)
    mono_APD = _is_monotonic_decreasing(APD)
    loss_apd = (1.0 - mono_APD) * 50.0

    # 5. Поощрение: F_amp на максимальной частоте > F_amp на минимальной
    f_amp_growth = float(F_amp[-1] - F_amp[0])
    bonus_growth = f_amp_growth * 1.0  # отрицательный вклад (поощрение)

    # 6. Поощрение: SR-load растёт с частотой (мера насыщения SR)
    sr_growth = float(SR[-1] - SR[0])
    bonus_sr = sr_growth * 20.0

    # Итог
    total = loss_mono_F + loss_tonus + loss_ca + loss_apd - bonus_growth - bonus_sr

    return {
        "loss_total": total,
        "loss_mono_F": loss_mono_F,
        "loss_tonus": loss_tonus,
        "loss_ca_overload": loss_ca,
        "loss_apd_mono": loss_apd,
        "bonus_F_growth": bonus_growth,
        "bonus_SR_growth": bonus_sr,
        "F_amp_min": float(F_amp.min()),
        "F_amp_max": float(F_amp.max()),
        "F_dia_max": float(F_dia.max()),
        "Ca_dia_max": ca_dia_max,
        "APD_min": float(APD.min()),
        "APD_max": float(APD.max()),
    }


# ---------------------------------------------------------------------------
# Графики
# ---------------------------------------------------------------------------


def plot_metric_grid(
    points: list[SweepPoint],
    biomarkers: list[dict],
    metric: str,
    metric_label: str,
    output_path: Path,
    title: str,
) -> None:
    """3×3 сетка графиков: для каждой комбинации (Vmax, K_NaCa) — кривая metric vs freq."""
    Vmax_vals = sorted({p.Vmax_up for p in points})
    KNaCa_vals = sorted({p.K_NaCa for p in points})

    n_v = len(Vmax_vals)
    n_k = len(KNaCa_vals)

    fig, axes = plt.subplots(
        n_v, n_k, figsize=(4 * n_k, 3 * n_v), sharex=True, sharey=True
    )
    if n_v == 1 and n_k == 1:
        axes = np.array([[axes]])
    elif n_v == 1:
        axes = axes[np.newaxis, :]
    elif n_k == 1:
        axes = axes[:, np.newaxis]

    # Группируем биомаркеры по (Vmax, K_NaCa)
    by_pair: dict[tuple, list] = {}
    for pt, b in zip(points, biomarkers):
        by_pair.setdefault((pt.Vmax_up, pt.K_NaCa), []).append((pt.frequency_Hz, b))

    for i, vm in enumerate(Vmax_vals):
        for j, kn in enumerate(KNaCa_vals):
            ax = axes[i, j]
            pairs = sorted(by_pair.get((vm, kn), []), key=lambda x: x[0])
            if not pairs:
                ax.set_visible(False)
                continue
            freqs = [p[0] for p in pairs]
            vals = [p[1][metric] for p in pairs]
            ax.plot(freqs, vals, "o-", color="#2c7bb6", lw=1.5, ms=6)
            ax.grid(alpha=0.3)
            ax.set_title(
                f"Vmax={vm*1e3:.2f}·10⁻³, K_NaCa={kn:.0f}",
                fontsize=9,
            )
            if i == n_v - 1:
                ax.set_xlabel("Частота, Гц", fontsize=9)
            if j == 0:
                ax.set_ylabel(metric_label, fontsize=9)
            ax.tick_params(labelsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    fig.suptitle(title, fontsize=12, y=1.00)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  → {output_path}")


def plot_best_overlay(
    points: list[SweepPoint],
    biomarkers: list[dict],
    rankings: list[dict],
    n_best: int,
    output_path: Path,
) -> None:
    """Лучшие N комбинаций vs baseline на 4 ключевых биомаркерах."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()

    metrics = [
        ("F_systolic_mN", "F_systolic, мН", axes[0]),
        ("F_amplitude_mN", "F_amplitude, мН", axes[1]),
        ("Ca_diastolic_nM", "Ca_dia, нМ", axes[2]),
        ("APD90_ms", "APD₉₀, мс", axes[3]),
    ]

    by_pair: dict[tuple, list] = {}
    for pt, b in zip(points, biomarkers):
        by_pair.setdefault((pt.Vmax_up, pt.K_NaCa), []).append((pt.frequency_Hz, b))

    # Цвета: tab10
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # Берём лучшие n_best и baseline (Vmax=5.8e-4, K_NaCa=5000)
    best_pairs = []
    for r in rankings[:n_best]:
        best_pairs.append((r["Vmax_up"], r["K_NaCa"]))

    baseline = (DEFAULT_PARAMS.Vmax_up, DEFAULT_PARAMS.K_NaCa)
    plot_pairs = [baseline] + [p for p in best_pairs if p != baseline]

    for metric_key, ylabel, ax in metrics:
        for idx, pair in enumerate(plot_pairs):
            data = sorted(by_pair.get(pair, []), key=lambda x: x[0])
            if not data:
                continue
            freqs = [d[0] for d in data]
            vals = [d[1][metric_key] for d in data]
            label = (
                "baseline"
                if pair == baseline
                else f"V={pair[0]*1e3:.2f}, K={pair[1]:.0f}"
            )
            color = "black" if pair == baseline else colors[idx]
            lw = 2.0 if pair == baseline else 1.4
            ls = "--" if pair == baseline else "-"
            ax.plot(
                freqs, vals, "o-", color=color, lw=lw, ms=5, linestyle=ls, label=label
            )
        ax.set_xlabel("Частота, Гц", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="best")
        ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(f"Лучшие {n_best} комбинаций vs baseline", fontsize=12, y=1.00)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  → {output_path}")


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


def save_biomarkers_csv(
    points: list[SweepPoint],
    biomarkers: list[dict],
    path: Path,
) -> None:
    """Полная таблица: все точки sweep × все биомаркеры."""
    if not biomarkers:
        return
    extra_keys = ["Vmax_up", "K_NaCa"]
    bio_keys = list(biomarkers[0].keys())
    fieldnames = extra_keys + bio_keys

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for pt, b in zip(points, biomarkers):
            row = {"Vmax_up": pt.Vmax_up, "K_NaCa": pt.K_NaCa}
            row.update(b)
            w.writerow(row)


def save_ranking_csv(rankings: list[dict], path: Path) -> None:
    if not rankings:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rankings[0].keys()))
        w.writeheader()
        for r in rankings:
            w.writerow(r)


def log_ranking_table(rankings: list[dict], top_n: int = 5) -> None:
    """Топ комбинаций в лог."""
    keys = [
        "Vmax_up",
        "K_NaCa",
        "loss_total",
        "F_amp_min",
        "F_amp_max",
        "F_dia_max",
        "Ca_dia_max",
    ]
    header = " | ".join(f"{k:>14}" for k in keys)
    logger.info("=" * len(header))
    logger.info(header)
    logger.info("-" * len(header))
    for r in rankings[:top_n]:
        row = " | ".join(
            f"{r[k]:>14.4g}" if isinstance(r[k], (int, float)) else f"{r[k]:>14}"
            for k in keys
        )
        logger.info(row)
    logger.info("=" * len(header))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sweep по Vmax_up × K_NaCa для подбора FFR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--vmax-up",
        default=",".join(f"{x:.6g}" for x in DEFAULT_VMAX_UP_VALUES),
        help="Значения Vmax_up через запятую (мМ/мс)",
    )
    parser.add_argument(
        "--k-naca",
        default=",".join(f"{x:.6g}" for x in DEFAULT_K_NACA_VALUES),
        help="Значения K_NaCa через запятую (пА/пФ)",
    )
    parser.add_argument(
        "--frequencies",
        default=",".join(f"{x:.1f}" for x in DEFAULT_FREQUENCIES),
        help="Частоты в Гц через запятую",
    )
    parser.add_argument("--warmup-cycles", type=int, default=300)
    parser.add_argument("--record-cycles", type=int, default=2)
    parser.add_argument(
        "--jobs", "-j", type=int, default=1, help="Число параллельных процессов"
    )
    parser.add_argument("--atol", type=float, default=1e-9)
    parser.add_argument("--rtol", type=float, default=1e-9)
    parser.add_argument("--max-step", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=Path("data/ffr_sweep"))
    parser.add_argument(
        "--n-best",
        type=int,
        default=3,
        help="Сколько лучших комбинаций показать на overlay-графике",
    )
    args = parser.parse_args(argv)

    Vmax_vals = [float(x) for x in args.vmax_up.split(",")]
    KNaCa_vals = [float(x) for x in args.k_naca.split(",")]
    freqs = [float(x) for x in args.frequencies.split(",")]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Сетка точек
    points = [
        SweepPoint(Vmax_up=vm, K_NaCa=kn, frequency_Hz=f)
        for vm, kn, f in itertools.product(Vmax_vals, KNaCa_vals, freqs)
    ]

    logger.info("=" * 60)
    logger.info("FFR sweep: подбор кальциевых параметров")
    logger.info(f"Vmax_up:     {Vmax_vals}")
    logger.info(f"K_NaCa:      {KNaCa_vals}")
    logger.info(f"Частоты:     {freqs} Гц")
    logger.info(f"Всего точек: {len(points)}")
    logger.info(f"Warmup:      {args.warmup_cycles} циклов")
    logger.info(f"n_jobs:      {args.jobs}")
    logger.info(f"Выход:       {args.output_dir}")
    logger.info("=" * 60)

    # ── Прогоны ─────────────────────────────────────────────────────────────
    t_total = time.perf_counter()

    def _one(pt: SweepPoint) -> SimulationResult:
        t0 = time.perf_counter()
        r = run_sweep_point(
            pt,
            n_cycles_warmup=args.warmup_cycles,
            n_cycles_record=args.record_cycles,
            atol=args.atol,
            rtol=args.rtol,
            max_step=args.max_step,
        )
        elapsed = time.perf_counter() - t0
        logger.info(
            f"  V={pt.Vmax_up*1e3:.2f}·10⁻³, K={pt.K_NaCa:.0f}, "
            f"f={pt.frequency_Hz:.1f} Гц → {elapsed:.1f} с"
        )
        return r

    if args.jobs == 1:
        results = [_one(pt) for pt in points]
    else:
        try:
            from joblib import Parallel, delayed
        except ImportError:
            logger.warning("joblib не установлен — n_jobs=1")
            results = [_one(pt) for pt in points]
        else:
            results = Parallel(n_jobs=args.jobs)(delayed(_one)(pt) for pt in points)

    elapsed_total = time.perf_counter() - t_total
    logger.success(
        f"Все прогоны: {elapsed_total/60:.1f} мин "
        f"({elapsed_total/len(points):.1f} с/прогон)"
    )

    # ── Биомаркеры ──────────────────────────────────────────────────────────
    biomarkers = [extract_biomarkers(r) for r in results]
    save_biomarkers_csv(points, biomarkers, args.output_dir / "biomarkers_all.csv")
    logger.info(f"Биомаркеры → {args.output_dir / 'biomarkers_all.csv'}")

    # ── Ранжирование комбинаций ─────────────────────────────────────────────
    by_pair: dict[tuple, list] = {}
    for pt, b in zip(points, biomarkers):
        by_pair.setdefault((pt.Vmax_up, pt.K_NaCa), []).append((pt.frequency_Hz, b))

    rankings = []
    for (vm, kn), pairs in by_pair.items():
        pairs_sorted = sorted(pairs, key=lambda x: x[0])
        bios_per_freq = [p[1] for p in pairs_sorted]
        loss_dict = compute_loss(bios_per_freq)
        rankings.append(
            {
                "Vmax_up": vm,
                "K_NaCa": kn,
                **loss_dict,
            }
        )

    rankings.sort(key=lambda x: x["loss_total"])
    save_ranking_csv(rankings, args.output_dir / "ranking.csv")
    logger.info(f"Ранжирование → {args.output_dir / 'ranking.csv'}")
    logger.info("")
    logger.info(f"ТОП-5 комбинаций (меньше loss = лучше):")
    log_ranking_table(rankings, top_n=5)

    # ── Сохраняем все прогоны ───────────────────────────────────────────────
    save_batch(results, args.output_dir / "runs.h5")
    logger.info(f"Прогоны → {args.output_dir / 'runs.h5'}")

    # ── Графики ─────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("Построение графиков:")
    plot_metric_grid(
        points,
        biomarkers,
        "F_systolic_mN",
        "F_systolic, мН",
        args.output_dir / "sweep_F_systolic.png",
        "F_systolic vs частота",
    )
    plot_metric_grid(
        points,
        biomarkers,
        "F_amplitude_mN",
        "F_amplitude, мН",
        args.output_dir / "sweep_F_amplitude.png",
        "F_amplitude vs частота",
    )
    plot_metric_grid(
        points,
        biomarkers,
        "Ca_diastolic_nM",
        "Ca_dia, нМ",
        args.output_dir / "sweep_Ca_diastolic.png",
        "Ca_diastolic vs частота",
    )
    plot_metric_grid(
        points,
        biomarkers,
        "APD90_ms",
        "APD₉₀, мс",
        args.output_dir / "sweep_APD90.png",
        "APD₉₀ vs частота",
    )
    plot_best_overlay(
        points, biomarkers, rankings, args.n_best, args.output_dir / "best_overlay.png"
    )

    logger.info("")
    logger.info("Что смотреть:")
    logger.info(f"  1. ranking.csv — топ комбинаций по loss")
    logger.info(f"  2. best_overlay.png — лучшие N кривых vs baseline")
    logger.info(f"  3. sweep_*.png — как каждый биомаркер реагирует на Vmax×K_NaCa")
    logger.info("")
    logger.info("Если ни одна комбинация не даёт хороший FFR — ")
    logger.info(
        "расширяй диапазон Vmax_up или добавляй другие параметры (V_rel, V_leak, Buf_sr)"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
