#!/usr/bin/env python
"""
scripts/ffr_diagnostic.py
=========================
Диагностика Force-Frequency Relationship (FFR) базовой модели.

Запускает прогоны на нескольких частотах, дожидается stationary,
извлекает биомаркеры на последнем цикле и строит FFR-кривые.

Используется для оценки текущего состояния модели ПЕРЕД подбором параметров.

Запуск:
    python scripts/ffr_diagnostic.py
    python scripts/ffr_diagnostic.py --warmup-cycles 500 --jobs 4
    python scripts/ffr_diagnostic.py --frequencies 0.5,1.0,2.0,3.0

Результат:
    data/ffr_diagnostic/
        runs.h5                    — все прогоны (save_batch)
        biomarkers.csv             — таблица биомаркеров
        biomarkers.png             — сводный график 4×2
        last_cycles_overview.png   — V/Ca/F на последнем цикле для всех частот
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

# sys.path: корень проекта и src/
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

# ---------------------------------------------------------------------------
# Биомаркеры
# ---------------------------------------------------------------------------


def _last_cycle_indices(t: np.ndarray, stim_period: float) -> tuple[int, int]:
    """Индексы границ последнего полного цикла."""
    t_end = t[-1]
    t_start = t_end - stim_period
    i0 = int(np.searchsorted(t, t_start, side="left"))
    return i0, len(t)


def _apd90(t: np.ndarray, V: np.ndarray, stim_start_in_cycle: float) -> float:
    """APD₉₀ — длительность ПД на 90% реполяризации.

    Считает от момента стимула (V начинает расти) до пересечения
    V_90 = V_min + 0.1 * (V_max - V_min) на спаде.
    """
    V_min = V.min()
    V_max = V.max()
    threshold = V_min + 0.1 * (V_max - V_min)

    # Время пика (момент после которого начинается реполяризация)
    i_peak = int(np.argmax(V))
    if i_peak >= len(V) - 1:
        return float("nan")

    # На спаде: первое пересечение порога
    repolar = V[i_peak:] - threshold
    crossings = np.where(np.diff(np.sign(repolar)))[0]
    if len(crossings) == 0:
        return float("nan")
    i_repol = i_peak + int(crossings[0])

    # APD = время реполяризации − время начала стимула
    t_start = t[0] + stim_start_in_cycle
    return float(t[i_repol] - t_start)


def _decay_tau(t: np.ndarray, ca: np.ndarray) -> float:
    """τ спада Ca_i: экспоненциальный фит на участке от пика до 50% спада."""
    i_peak = int(np.argmax(ca))
    if i_peak >= len(ca) - 5:
        return float("nan")

    ca_peak = ca[i_peak]
    ca_min = ca[-1]  # диастолический в конце цикла
    ca_50 = ca_min + 0.5 * (ca_peak - ca_min)

    decay = ca[i_peak:]
    below = np.where(decay <= ca_50)[0]
    if len(below) == 0:
        return float("nan")

    i_50 = i_peak + int(below[0])
    if i_50 - i_peak < 3:
        return float("nan")

    # Линейный фит log(ca - ca_min) ~ -t/tau
    seg_t = t[i_peak:i_50] - t[i_peak]
    seg_ca = ca[i_peak:i_50] - ca_min
    if np.any(seg_ca <= 0):
        return float("nan")
    log_ca = np.log(seg_ca)
    slope, _ = np.polyfit(seg_t, log_ca, 1)
    if slope >= 0:
        return float("nan")
    return float(-1.0 / slope)


def extract_biomarkers(result: SimulationResult) -> dict[str, float]:
    """Извлечь 8 биомаркеров из последнего цикла прогона."""
    period = result.params.stim_period
    i0, i1 = _last_cycle_indices(result.time, period)

    t = result.time[i0:i1] - result.time[i0]
    V = result.V[i0:i1]
    Ca = result.Ca_i[i0:i1]
    F = result.F_XSE[i0:i1]
    Ca_nSR = result.variables["Ca_nSR"][i0:i1]

    # Время пика Ca от стимула (стимул в начале цикла)
    stim_start = result.params.stim_start
    i_peak_ca = int(np.argmax(Ca))
    t_to_peak = float(t[i_peak_ca] - stim_start)

    return {
        "frequency_Hz": 1000.0 / period,
        "stim_period_ms": period,
        "APD90_ms": _apd90(t, V, stim_start),
        "Ca_amplitude_uM": float((Ca.max() - Ca.min()) * 1e3),
        "Ca_diastolic_nM": float(Ca.min() * 1e6),
        "Ca_systolic_uM": float(Ca.max() * 1e3),
        "Ca_time_to_peak_ms": t_to_peak,
        "Ca_decay_tau_ms": _decay_tau(t, Ca),
        "F_systolic_mN": float(F.max()),
        "F_diastolic_mN": float(F.min()),
        "F_amplitude_mN": float(F.max() - F.min()),
        "Ca_nSR_mean_mM": float(Ca_nSR.mean()),
        "Ca_nSR_min_mM": float(Ca_nSR.min()),
        "V_resting_mV": float(V.min()),
        "V_peak_mV": float(V.max()),
    }


# ---------------------------------------------------------------------------
# Запуск одного прогона
# ---------------------------------------------------------------------------


def run_at_frequency(
    freq_hz: float,
    base_params: TNNPMParams,
    n_cycles_warmup: int,
    n_cycles_record: int,
    atol: float = 1e-9,
    rtol: float = 1e-9,
    max_step: float = 0.5,
) -> SimulationResult:
    """Один прогон: warmup + запись.

    Записываем сразу всё (warmup + record). Биомаркеры берутся с последнего цикла.
    Это проще чем делать два прогона и подсасывать last_state.
    """
    period = 1000.0 / freq_hz
    p = base_params.replace(stim_period=period)

    total_ms = (n_cycles_warmup + n_cycles_record) * period

    # n_out: ~2000 точек на последний цикл достаточно для биомаркеров
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

    t_wall_0 = time.perf_counter()
    result = run_single(p, state0, config, l0=l0)
    elapsed = time.perf_counter() - t_wall_0

    logger.info(
        f"  {freq_hz:.2f} Гц: {n_cycles_warmup}+{n_cycles_record} циклов = "
        f"{total_ms/1000:.0f} с симуляции за {elapsed:.1f} с реального времени"
    )
    return result


# ---------------------------------------------------------------------------
# Графики
# ---------------------------------------------------------------------------


def plot_biomarkers_vs_frequency(
    biomarkers: list[dict],
    output_path: Path,
) -> None:
    """Сводный график 4×2: биомаркер vs частота."""
    freq = np.array([b["frequency_Hz"] for b in biomarkers])

    panels = [
        ("F_systolic_mN", "Пиковая сила, мН", "F_XSE peak"),
        ("F_amplitude_mN", "Амплитуда силы, мН", "F_XSE amplitude"),
        ("Ca_amplitude_uM", "Амплитуда [Ca²⁺]ᵢ, мкМ", "Ca transient"),
        ("Ca_systolic_uM", "Пиковый [Ca²⁺]ᵢ, мкМ", "Ca peak"),
        ("Ca_diastolic_nM", "Диастолический [Ca²⁺]ᵢ, нМ", "Ca diastolic"),
        ("APD90_ms", "APD₉₀, мс", "APD"),
        ("Ca_decay_tau_ms", "τ спада Ca, мс", "Ca decay"),
        ("Ca_nSR_mean_mM", "Средний Ca в nSR, мМ", "SR load"),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(11, 12))
    axes = axes.flatten()

    for ax, (key, ylabel, title) in zip(axes, panels):
        vals = np.array([b[key] for b in biomarkers])
        ax.plot(freq, vals, "o-", color="#2c7bb6", lw=1.4, ms=6)
        ax.set_xlabel("Частота, Гц", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Force-Frequency Relationship — диагностика базовой модели",
        fontsize=12,
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"График сохранён: {output_path}")


def plot_last_cycles_overlay(
    results: list[SimulationResult],
    output_path: Path,
) -> None:
    """V, Ca_i, F_XSE на последнем цикле — все частоты на одной фигуре."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=False)

    cmap = plt.cm.viridis
    n = len(results)
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

    for r, color in zip(results, colors):
        period = r.params.stim_period
        i0, i1 = _last_cycle_indices(r.time, period)
        t = r.time[i0:i1] - r.time[i0]
        freq = 1000.0 / period
        label = f"{freq:.1f} Гц"

        axes[0].plot(t, r.V[i0:i1], color=color, lw=1.0, label=label)
        axes[1].plot(t, r.Ca_i[i0:i1] * 1e3, color=color, lw=1.0, label=label)
        axes[2].plot(t, r.F_XSE[i0:i1], color=color, lw=1.0, label=label)

    axes[0].set_ylabel("V, мВ")
    axes[0].set_title("Потенциал действия (последний цикл)")
    axes[0].legend(fontsize=8, loc="upper right", ncol=2, frameon=False)

    axes[1].set_ylabel("[Ca²⁺]ᵢ, мкМ")
    axes[1].set_title("Кальций")

    axes[2].set_ylabel("F_XSE, мН")
    axes[2].set_xlabel("Время от начала цикла, мс")
    axes[2].set_title("Сила")

    for ax in axes:
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"График сохранён: {output_path}")


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


def save_biomarkers_csv(biomarkers: list[dict], path: Path) -> None:
    if not biomarkers:
        return
    fieldnames = list(biomarkers[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in biomarkers:
            writer.writerow(row)
    logger.info(f"Биомаркеры сохранены: {path}")


def log_biomarkers_table(biomarkers: list[dict]) -> None:
    """Вывод таблицы биомаркеров в логи."""
    keys = [
        "frequency_Hz",
        "F_amplitude_mN",
        "Ca_amplitude_uM",
        "Ca_diastolic_nM",
        "APD90_ms",
        "Ca_decay_tau_ms",
        "Ca_nSR_mean_mM",
    ]

    header = " | ".join(f"{k:>16}" for k in keys)
    logger.info("=" * len(header))
    logger.info(header)
    logger.info("-" * len(header))
    for b in biomarkers:
        row = " | ".join(f"{b[k]:>16.4g}" for k in keys)
        logger.info(row)
    logger.info("=" * len(header))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Диагностика FFR базовой модели TNNPM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--frequencies",
        default="0.5,1.0,1.5,2.0,2.5,3.0",
        help="Частоты в Гц через запятую (по умолчанию: 0.5,1.0,1.5,2.0,2.5,3.0)",
    )
    parser.add_argument(
        "--warmup-cycles",
        type=int,
        default=300,
        help="Сколько циклов на выход на stationary (по умолчанию: 300)",
    )
    parser.add_argument(
        "--record-cycles",
        type=int,
        default=2,
        help="Сколько циклов записать после warmup (по умолчанию: 2; "
        "биомаркеры берутся с последнего)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/ffr_diagnostic"),
        help="Директория для результатов",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-9,
        help="Абсолютная точность CVode",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-9,
        help="Относительная точность CVode",
    )
    parser.add_argument(
        "--max-step",
        type=float,
        default=0.5,
        help="Макс. шаг CVode, мс",
    )
    args = parser.parse_args(argv)

    # ── Подготовка ──────────────────────────────────────────────────────────
    frequencies = [float(x) for x in args.frequencies.split(",")]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Диагностика Force-Frequency Relationship")
    logger.info(f"Частоты:        {frequencies} Гц")
    logger.info(f"Warmup:         {args.warmup_cycles} циклов")
    logger.info(f"Запись:         {args.record_cycles} циклов")
    logger.info(f"Допуски CVode:  atol={args.atol:.0e}, rtol={args.rtol:.0e}")
    logger.info(f"Выход:          {args.output_dir}")
    logger.info("=" * 60)

    # ── Прогоны ─────────────────────────────────────────────────────────────
    t_total = time.perf_counter()
    results: list[SimulationResult] = []
    for f in frequencies:
        r = run_at_frequency(
            f,
            DEFAULT_PARAMS,
            n_cycles_warmup=args.warmup_cycles,
            n_cycles_record=args.record_cycles,
            atol=args.atol,
            rtol=args.rtol,
            max_step=args.max_step,
        )
        results.append(r)

    elapsed_total = time.perf_counter() - t_total
    logger.success(
        f"Все прогоны завершены за {elapsed_total/60:.1f} мин "
        f"({elapsed_total/len(frequencies):.1f} с/прогон)"
    )

    # ── Биомаркеры ──────────────────────────────────────────────────────────
    biomarkers = [extract_biomarkers(r) for r in results]
    log_biomarkers_table(biomarkers)

    # ── Сохранение ──────────────────────────────────────────────────────────
    save_batch(results, args.output_dir / "runs.h5")
    logger.info(f"Прогоны сохранены: {args.output_dir / 'runs.h5'}")

    save_biomarkers_csv(biomarkers, args.output_dir / "biomarkers.csv")

    # ── Графики ─────────────────────────────────────────────────────────────
    plot_biomarkers_vs_frequency(biomarkers, args.output_dir / "biomarkers.png")
    plot_last_cycles_overlay(results, args.output_dir / "last_cycles_overlay.png")

    # ── Краткий вывод ───────────────────────────────────────────────────────
    logger.info("")
    logger.info("Что смотреть на графиках:")
    logger.info("  F_systolic_mN ↑ с частотой → позитивный FFR (норма)")
    logger.info("  F_systolic_mN ↓ с частотой → негативный FFR (патология)")
    logger.info("  APD₉₀ ↓ с частотой         → корректная rate adaptation")
    logger.info(
        "  Ca_diastolic ↑ с частотой  → накопление Ca (характерно для тахикардии)"
    )
    logger.info(
        "  Ca_nSR_mean ↑ с частотой   → растёт нагрузка SR (источник позитивного FFR)"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
