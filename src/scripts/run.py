#!/usr/bin/env python
"""
scripts/run.py
==============
Точка входа CLI для запуска симуляций TNNPM.

Использование:
    # Одиночный прогон с параметрами по умолчанию
    python scripts/run.py

    # Прогон из конфига
    python scripts/run.py --config configs/default.yaml

    # Sweep из конфига, 4 параллельных процесса
    python scripts/run.py --config configs/2Hz_sweep_gCaL.yaml --jobs 4

    # Переопределить отдельный параметр прямо из командной строки
    python scripts/run.py --config configs/default.yaml --set g_Na=16.0

    # Задать warmup-файл (начальные условия из предыдущего прогона)
    python scripts/run.py --config configs/default.yaml \\
                          --warmup data/warmup_1Hz.h5 \\
                          --output data/2Hz_1000c.h5

    # Построить графики после прогона
    python scripts/run.py --config configs/default.yaml --plot

Выходной файл:
    Одиночный прогон → HDF5 через simulation_io.save()
    Sweep            → HDF5 через simulation_io.save_batch()
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Добавляем корень проекта в sys.path чтобы импорты работали
# независимо от того, откуда запущен скрипт
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from loguru import logger

from experiment import run_single, run_sweep, sweep_combinations
from parameters import (
    DEFAULT_CONFIG,
    DEFAULT_PARAMS,
    InitialState,
    SimConfig,
    TNNPMParams,
)
from simulation_io import load_state, save, save_batch

# ---------------------------------------------------------------------------
# Разбор аргументов
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Запуск симуляции модели TNNPM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--config",
        "-c",
        metavar="PATH",
        help="Путь к YAML-конфигу (configs/default.yaml)",
    )
    p.add_argument(
        "--output",
        "-o",
        metavar="PATH",
        default=None,
        help="Путь для сохранения результата (.h5). "
        "По умолчанию: data/<config_name>.h5",
    )
    p.add_argument(
        "--warmup",
        "-w",
        metavar="PATH",
        default=None,
        help="HDF5-файл с начальными условиями (последняя точка предыдущего прогона)",
    )
    p.add_argument(
        "--set",
        "-s",
        metavar="KEY=VALUE",
        action="append",
        default=[],
        dest="overrides",
        help="Переопределить параметр: --set g_Na=16.0 --set stim_period=500",
    )
    p.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=1,
        metavar="N",
        help="Число параллельных процессов для sweep (по умолчанию: 1)",
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="Построить сводный график после прогона (требует matplotlib)",
    )
    p.add_argument(
        "--plot-last",
        type=float,
        default=None,
        metavar="MS",
        help="Показывать только последние N мс на графике",
    )
    p.add_argument(
        "--no-save",
        action="store_true",
        help="Не сохранять результат на диск (для отладки)",
    )
    return p


# ---------------------------------------------------------------------------
# Загрузка конфига
# ---------------------------------------------------------------------------


def _load_yaml(path: str) -> dict:
    try:
        import yaml
    except ImportError:
        logger.error("PyYAML не установлен. Установите: pip install pyyaml")
        sys.exit(1)
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _parse_overrides(overrides: list[str]) -> dict:
    """Разобрать ['g_Na=16.0', 'stim_period=500'] → {'g_Na': 16.0, ...}"""
    result = {}
    for item in overrides:
        if "=" not in item:
            logger.error(f"Неверный формат --set: {item!r}. Ожидается KEY=VALUE")
            sys.exit(1)
        key, _, raw = item.partition("=")
        key = key.strip()
        # Пробуем int → float → str
        for cast in (int, float, str):
            try:
                result[key] = cast(raw)
                break
            except ValueError:
                continue
    return result


def _build_params_and_config(
    yaml_data: dict,
    overrides: dict,
) -> tuple[TNNPMParams | dict[str, list], SimConfig, dict]:
    """Из YAML + overrides собрать params (или sweep-dict), config, mechanics."""
    raw_params = dict(yaml_data.get("params", {}))
    raw_config = dict(yaml_data.get("config", {}))
    raw_mechanics = dict(yaml_data.get("mechanics", {}))

    # Применяем CLI-overrides поверх YAML
    for k, v in overrides.items():
        raw_params[k] = v

    # Определяем: sweep или одиночный прогон?
    sweep_keys = {k: v for k, v in raw_params.items() if isinstance(v, list)}
    scalar_params = {k: v for k, v in raw_params.items() if not isinstance(v, list)}

    base_params = TNNPMParams.from_dict(scalar_params)
    config = SimConfig.from_dict(raw_config)

    if sweep_keys:
        return base_params, sweep_keys, config, raw_mechanics
    return base_params, None, config, raw_mechanics


# ---------------------------------------------------------------------------
# Вывод информации перед прогоном
# ---------------------------------------------------------------------------


def _log_run_info(
    params,
    sweep: dict | None,
    config: SimConfig,
    state0: InitialState,
    output_path: Path,
) -> None:
    duration_s = config.time_stop / 1000.0
    cycles = config.time_stop / params.stim_period

    logger.info("=" * 55)
    if sweep:
        n_runs = 1
        for v in sweep.values():
            n_runs *= len(v)
        logger.info(f"Режим: sweep ({n_runs} прогонов)")
        for k, vals in sweep.items():
            logger.info(f"  {k}: {vals}")
    else:
        logger.info("Режим: одиночный прогон")

    logger.info(f"Частота стимуляции: {1000/params.stim_period:.2f} Гц")
    logger.info(f"Длительность: {duration_s:.0f} с = {cycles:.0f} циклов")
    logger.info(f"Допуски CVode: atol={config.atol:.0e}  rtol={config.rtol:.0e}")
    logger.info(f"V₀ = {state0.V:.2f} мВ  |  Ca_i₀ = {state0.Ca_i:.2e} мМ")
    logger.info(f"Выход: {output_path}")
    logger.info("=" * 55)


# ---------------------------------------------------------------------------
# Графики
# ---------------------------------------------------------------------------


def _make_plots(results, plot_last: float | None, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        from plots import plot_overview, plot_sweep
    except ImportError:
        logger.warning("matplotlib не установлен — графики пропущены")
        return

    figures_dir = output_path.parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(results, list):
        # Sweep: sweep-графики по первому варьируемому параметру
        if len(results) == 0:
            return
        # Определяем параметр sweep по различиям между результатами
        sweep_param = _detect_sweep_param(results)
        for var in ["V", "Ca_i"]:
            try:
                fig, _ = plot_sweep(results, var, sweep_param, last_n_ms=plot_last)
                out = figures_dir / f"sweep_{var}.png"
                fig.savefig(out, dpi=150, bbox_inches="tight")
                logger.info(f"График сохранён: {out}")
                plt.close(fig)
            except Exception as e:
                logger.warning(f"Не удалось построить sweep-график {var}: {e}")
    else:
        # Одиночный прогон
        fig, _ = plot_overview(results, last_n_ms=plot_last, title=output_path.stem)
        out = figures_dir / f"{output_path.stem}_overview.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        logger.info(f"График сохранён: {out}")
        plt.close(fig)


def _detect_sweep_param(results) -> str:
    """Найти первый параметр который различается между прогонами."""
    if len(results) < 2:
        return "g_CaL"
    import dataclasses

    r0, r1 = results[0], results[1]
    for fld in dataclasses.fields(r0.params):
        if getattr(r0.params, fld.name) != getattr(r1.params, fld.name):
            return fld.name
    return "g_CaL"


# ---------------------------------------------------------------------------
# Главная функция
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # ── Загрузка конфига ────────────────────────────────────────────────────
    yaml_data = _load_yaml(args.config) if args.config else {}
    overrides = _parse_overrides(args.overrides)
    base_params, sweep, config, mechanics = _build_params_and_config(
        yaml_data, overrides
    )
    F_afterload = float(mechanics.get("F_afterload", 0.0))

    # ── Начальные условия ───────────────────────────────────────────────────
    warmup_file = args.warmup or yaml_data.get("warmup_file")
    if warmup_file:
        logger.info(f"Загрузка начальных условий из {warmup_file!r}")
        state0 = load_state(warmup_file)
    else:
        state0 = None  # run_single вызовет make_init_state автоматически

    # ── Путь вывода ─────────────────────────────────────────────────────────
    if args.output:
        output_path = Path(args.output)
    elif args.config:
        cfg_name = Path(args.config).stem
        output_path = Path("data") / f"{cfg_name}.h5"
    else:
        output_path = Path("data") / "result.h5"

    # ── Информация перед запуском ────────────────────────────────────────────
    _state_for_log = state0 or InitialState()
    _log_run_info(base_params, sweep, config, _state_for_log, output_path)

    # ── Запуск ──────────────────────────────────────────────────────────────
    t0 = time.perf_counter()

    if sweep:
        logger.info("Запуск sweep...")
        results = run_sweep(
            sweep=sweep,
            base_params=base_params,
            state0=state0,
            config=config,
            F_afterload=F_afterload,
            n_jobs=args.jobs,
            recompute_init_mechanics=True,
        )
        elapsed = time.perf_counter() - t0
        logger.success(
            f"Sweep завершён: {len(results)} прогонов за {elapsed:.1f} с "
            f"({elapsed/len(results):.1f} с/прогон)"
        )
        if not args.no_save:
            save_batch(results, output_path)
            logger.info(f"Результаты сохранены → {output_path}")
        if args.plot:
            _make_plots(results, args.plot_last, output_path)
    else:
        logger.info("Запуск прогона...")
        result = run_single(
            params=base_params,
            state0=state0,
            config=config,
            F_afterload=F_afterload,
            recompute_init_mechanics=(state0 is not None),
        )
        elapsed = time.perf_counter() - t0
        logger.success(
            f"Прогон завершён за {elapsed:.1f} с  " f"({result.meta['n_points']} точек)"
        )
        if not args.no_save:
            save(result, output_path)
            logger.info(f"Результат сохранён → {output_path}")
        if args.plot:
            _make_plots(result, args.plot_last, output_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
