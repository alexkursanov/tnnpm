"""
experiment.py
=============
Высокоуровневый API для запуска симуляций.

Публичный API:
    run_single(params, state0, config, ...) -> SimulationResult
        Один прогон. Основной строительный блок для всего остального.

    run_sweep(base_params, sweep, state0, config, ...) -> list[SimulationResult]
        Параметрический sweep по сетке значений. Параллельный запуск.

    SimulationResult
        Иммутабельный dataclass с результатами одного прогона.

Пример — одиночный прогон:
    result = run_single(DEFAULT_PARAMS, state0, DEFAULT_CONFIG)
    plt.plot(result.time, result.variables['V'])

Пример — sweep по g_CaL:
    results = run_sweep(
        base_params=DEFAULT_PARAMS,
        sweep={'g_CaL': [3e-5, 5e-5, 7e-5]},
        state0=state0,
        config=DEFAULT_CONFIG,
        n_jobs=4,
    )

Пример — для оптимизатора (scipy / optuna):
    def objective(x):
        p = DEFAULT_PARAMS.replace(g_CaL=x[0], g_Na=x[1])
        r = run_single(p, state0, TEST_CONFIG)
        return loss(r)
"""

from __future__ import annotations

import dataclasses
import itertools
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from model import calculate_outputs, make_init_state, rhs
from parameters import (
    DEFAULT_CONFIG,
    DEFAULT_PARAMS,
    InitialState,
    SimConfig,
    TNNPMParams,
)
from solver import integrate

# ---------------------------------------------------------------------------
# SimulationResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimulationResult:
    """Результат одного прогона модели TNNPM.

    Все массивы имеют форму (N,) где N — число точек вывода.

    Поля:
        time       — время (мс), shape (N,)
        variables  — фазовые переменные, dict[str, ndarray(N,)]
        currents   — ионные токи, dict[str, ndarray(N,)]
        forces     — механические силы, dict[str, ndarray(N,)]
        params     — параметры этого прогона (TNNPMParams)
        state0     — начальные условия (InitialState)
        config     — конфигурация прогона (SimConfig)
        meta       — служебная информация (время счёта, версия и т.д.)
    """

    time: np.ndarray
    variables: dict[str, np.ndarray]
    currents: dict[str, np.ndarray]
    forces: dict[str, np.ndarray]
    params: TNNPMParams
    state0: InitialState
    config: SimConfig
    meta: dict[str, Any] = field(default_factory=dict)

    # Удобные свойства для частых обращений
    @property
    def V(self) -> np.ndarray:
        return self.variables["V"]

    @property
    def Ca_i(self) -> np.ndarray:
        return self.variables["Ca_i"]

    @property
    def F_XSE(self) -> np.ndarray:
        return self.forces["F_XSE"]

    def last_state(self) -> InitialState:
        """Последняя точка прогона как InitialState.

        Используется для цепочки прогонов:
            r1 = run_single(p, state0, config_warmup)
            r2 = run_single(p, r1.last_state(), config_main)
        """
        return InitialState.from_result(self.variables)


# ---------------------------------------------------------------------------
# Внутренние утилиты
# ---------------------------------------------------------------------------

_VAR_NAMES = [
    "d",
    "f2",
    "fCass",
    "f",
    "Ca_SR",
    "Ca_i",
    "Ca_ss",
    "p_iup",
    "h",
    "j",
    "m",
    "V",
    "K_i",
    "Xr1",
    "Xr2",
    "Xs",
    "Na_i",
    "r",
    "s",
    "v",
    "w",
    "N",
    "A",
    "l_1",
    "l_2",
    "l_3",
    "R",
    "O",
    "I",
    "RI",
    "CaMKt",
    "Ca_nSR",
    "Ca_jSR",
]


def _y_to_variables(y: np.ndarray) -> dict[str, np.ndarray]:
    """Матрица состояний (N, 33) → словарь именованных массивов."""
    return {name: y[:, i] for i, name in enumerate(_VAR_NAMES)}


def _make_rhs_fn(p: TNNPMParams, F_afterload: float, l0: float):
    """Замыкание rhs с зафиксированными параметрами для передачи в солвер."""

    def fn(t: float, y: np.ndarray) -> np.ndarray:
        return rhs(t, y, p, F_afterload=F_afterload, l0=l0)

    return fn


# ---------------------------------------------------------------------------
# run_single
# ---------------------------------------------------------------------------


def run_single(
    params: TNNPMParams = DEFAULT_PARAMS,
    state0: InitialState | None = None,
    config: SimConfig = DEFAULT_CONFIG,
    *,
    F_afterload: float = 0.0,
    l0: float | None = None,
    recompute_init_mechanics: bool = False,
) -> SimulationResult:
    """Запустить один прогон модели TNNPM.

    Аргументы:
        params       — параметры модели
        state0       — начальные условия. Если None, используется InitialState()
                       (стандартное стационарное состояние при 1 Гц).
        config       — параметры прогона (время, допуски солвера)
        F_afterload  — постнагрузка (мН). 0 = изометрический режим.
        l0           — референсная длина (мкм). Если None — вычисляется из state0.
        recompute_init_mechanics — если True, пересчитать механическое равновесие
                       через make_init_state() поверх переданного state0.
                       Нужно при смене механических параметров между прогонами.

    Возвращает:
        SimulationResult
    """
    t_start_wall = time.perf_counter()

    # Начальные условия
    if state0 is None:
        state0, l0_computed = make_init_state(params)
        if l0 is None:
            l0 = l0_computed
    elif recompute_init_mechanics:
        state0, l0_computed = make_init_state(params, base=state0)
        if l0 is None:
            l0 = l0_computed
    else:
        if l0 is None:
            l0 = state0.l_2 + state0.l_3

    y0 = np.array(state0.to_array(), dtype=np.float64)

    # Интегрирование
    rhs_fn = _make_rhs_fn(params, F_afterload, l0)
    t_arr, y_arr = integrate(rhs_fn, y0, config)

    # Постобработка
    variables = _y_to_variables(y_arr)
    outputs = calculate_outputs(t_arr, y_arr, params, F_afterload=F_afterload, l0=l0)

    currents = {
        k: outputs[k]
        for k in [
            "i_Na",
            "i_CaL",
            "i_NaCa",
            "i_NaK",
            "i_K1",
            "i_Kr",
            "i_Ks",
            "i_K_ATP",
            "i_to",
            "i_rel",
            "i_up",
            "i_leak",
            "i_xfer",
            "i_b_Ca",
            "i_p_Ca",
        ]
    }
    forces = {
        k: outputs[k] for k in ["F_CE", "F_SE", "F_PE", "F_VS1", "F_VS2", "F_XSE"]
    }

    meta = {
        "wall_time_s": time.perf_counter() - t_start_wall,
        "n_points": len(t_arr),
        "F_afterload": F_afterload,
        "l0": l0,
    }

    return SimulationResult(
        time=t_arr,
        variables=variables,
        currents=currents,
        forces=forces,
        params=params,
        state0=state0,
        config=config,
        meta=meta,
    )


# ---------------------------------------------------------------------------
# run_sweep
# ---------------------------------------------------------------------------


def run_sweep(
    sweep: dict[str, list],
    base_params: TNNPMParams = DEFAULT_PARAMS,
    state0: InitialState | None = None,
    config: SimConfig = DEFAULT_CONFIG,
    *,
    F_afterload: float = 0.0,
    l0: float | None = None,
    n_jobs: int = 1,
    recompute_init_mechanics: bool = False,
) -> list[SimulationResult]:
    """Параметрический sweep по сетке значений.

    Запускает run_single для каждой комбинации параметров из sweep.
    При n_jobs > 1 использует joblib для параллельного выполнения.

    Аргументы:
        sweep    — словарь {имя_параметра: [значение1, значение2, ...]}.
                   Перебираются все комбинации (декартово произведение).
        n_jobs   — число параллельных процессов. -1 = все доступные ядра.

    Пример:
        results = run_sweep(
            sweep={'g_CaL': [3e-5, 5e-5], 'stim_period': [500., 1000.]},
            base_params=DEFAULT_PARAMS,
            config=DEFAULT_CONFIG,
            n_jobs=4,
        )
        # 4 прогона: все комбинации g_CaL × stim_period

    Возвращает:
        Список SimulationResult в том же порядке что и комбинации параметров.
        Порядок: первый параметр меняется медленнее (как np.meshgrid indexing='ij').
    """
    # Строим список комбинаций
    param_names = list(sweep.keys())
    param_values = list(sweep.values())
    combinations = list(itertools.product(*param_values))

    def _make_params(combo: tuple) -> TNNPMParams:
        kwargs = dict(zip(param_names, combo))
        return base_params.replace(**kwargs)

    def _run_one(combo: tuple) -> SimulationResult:
        p = _make_params(combo)
        return run_single(
            params=p,
            state0=state0,
            config=config,
            F_afterload=F_afterload,
            l0=l0,
            recompute_init_mechanics=recompute_init_mechanics,
        )

    if n_jobs == 1:
        return [_run_one(c) for c in combinations]

    # Параллельный запуск через joblib
    try:
        from joblib import Parallel, delayed
    except ImportError as e:
        raise ImportError(
            "joblib не установлен. Установите: pip install joblib, "
            "или используйте n_jobs=1."
        ) from e

    return Parallel(n_jobs=n_jobs)(delayed(_run_one)(c) for c in combinations)


def sweep_combinations(sweep: dict[str, list]) -> list[dict]:
    """Вернуть список всех комбинаций sweep как список dict.

    Вспомогательная функция для предварительного просмотра sweep
    без запуска симуляций.

    Пример:
        for combo in sweep_combinations({'g_CaL': [3e-5, 5e-5], 'g_Na': [14., 16.]}):
            print(combo)
        # {'g_CaL': 3e-05, 'g_Na': 14.0}
        # {'g_CaL': 3e-05, 'g_Na': 16.0}
        # ...
    """
    names = list(sweep.keys())
    values = list(sweep.values())
    return [dict(zip(names, combo)) for combo in itertools.product(*values)]
