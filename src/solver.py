"""
solver.py
=========
Тонкая обёртка над CVode (Assimulo).

Публичный API:
    integrate(rhs_fn, y0, config) -> tuple[np.ndarray, np.ndarray]

Принципы:
    - Чистая функция: никакого глобального состояния.
    - Все настройки CVode приходят из SimConfig — не зашиты в код.
    - num_threads=1: параллелизм на уровне experiment.py (много прогонов),
      а не внутри одного прогона.
    - Исправлена ошибка оригинала: stop*1000 использовалось и как tfinal,
      и как ncp (число точек вывода) — теперь это два разных параметра.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from assimulo.problem import Explicit_Problem
from assimulo.solvers import CVode

from parameters import SimConfig


def integrate(
    rhs_fn: Callable[[float, np.ndarray], np.ndarray],
    y0: np.ndarray,
    config: SimConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Интегрировать систему ОДУ методом BDF (CVode).

    Аргументы:
        rhs_fn  — правая часть dy/dt = f(t, y).
                  Должна принимать (float, np.ndarray) и возвращать np.ndarray.
                  Типичный вызов: functools.partial(rhs, p=p, F_afterload=0., l0=l0)
        y0      — начальные условия (np.ndarray, float64)
        config  — параметры прогона и солвера (SimConfig)

    Возвращает:
        t  — массив времён (N,), мс
        y  — матрица состояний (N, n_states)
    """
    # Assimulo ожидает сигнатуру f(t, y) — оборачиваем
    problem = Explicit_Problem(rhs_fn, y0=y0.astype(np.float64), t0=config.time_start)
    problem.name = "TNNPM"

    sim = CVode(problem)

    # Метод интегрирования
    sim.iter = "Newton"
    sim.discr = "BDF"
    sim.maxord = 5
    sim.linear_solver = "DENSE"

    # Допуски
    sim.atol = config.atol
    sim.rtol = config.rtol

    # Шаг
    sim.maxh = config.max_step
    sim.minh = 0.0
    sim.inith = 0.0

    # Вывод: тихий режим
    sim.report_continuously = False
    sim.display_progress = False
    sim.verbosity = 50  # CRITICAL — подавляет INFO-сообщения CVode

    # Один поток: параллелизм через experiment.run_sweep, не внутри прогона
    sim.num_threads = 1

    # Число точек вывода:
    #   None → CVode сам выбирает шаги (только внутренние точки адаптивного шага)
    #   int  → равномерная сетка из n_out точек на [t0, tfinal]
    # CVode требует int для ncp; YAML может передать float (1000.0 вместо 1000)
    ncp = int(config.n_out) if config.n_out is not None else 0

    t, y = sim.simulate(config.time_stop, ncp)

    return np.array(t), np.array(y)
