"""
plots.py
========
Функции визуализации над SimulationResult.

Принципы:
    - Каждая функция создаёт свою Figure — нет глобального состояния plt.
      Повторные вызовы не накапливают линии на одном графике.
    - Возвращают (fig, ax) — вызывающий код решает: show() или savefig().
    - Параметры осей (заголовок, метки) выводятся из данных автоматически,
      но могут быть переопределены через keyword-аргументы.
    - Sweep: функции принимают список результатов и рисуют их на одном графике
      с автоматической легендой по варьируемому параметру.

Пример — одиночный прогон:
    fig, ax = plot_voltage(result)
    fig.savefig('reports/figures/voltage.png', dpi=150)

Пример — sweep:
    results = run_sweep(sweep={'g_CaL': [3e-5, 5e-5, 7e-5]}, ...)
    fig, ax = plot_sweep(results, variable='Ca_i', sweep_param='g_CaL')
    plt.show()
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from experiment import SimulationResult

# Цветовая схема для sweep-графиков (matplotlib tab10)
_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


# ---------------------------------------------------------------------------
# Одиночный прогон — основные переменные
# ---------------------------------------------------------------------------


def plot_voltage(
    result: SimulationResult,
    *,
    last_n_ms: float | None = None,
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Потенциал действия V(t).

    Аргументы:
        last_n_ms — если задан, показывать только последние N миллисекунд.
                    Удобно чтобы убрать переходный процесс.
    """
    fig, ax = _get_ax(ax)
    t, v = _slice_last(result.time, result.V, last_n_ms)
    ax.plot(t, v, color="#2c7bb6", linewidth=0.8)
    ax.set_xlabel("Время, мс")
    ax.set_ylabel("V, мВ")
    ax.set_title(title or "Потенциал действия")
    _apply_style(ax)
    return fig, ax


def plot_calcium(
    result: SimulationResult,
    *,
    last_n_ms: float | None = None,
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Внутриклеточный кальций Ca_i(t)."""
    fig, ax = _get_ax(ax)
    t, ca = _slice_last(result.time, result.Ca_i, last_n_ms)
    ax.plot(t, ca * 1e3, color="#d7191c", linewidth=0.8)  # мкМ
    ax.set_xlabel("Время, мс")
    ax.set_ylabel("[Ca²⁺]ᵢ, мкМ")
    ax.set_title(title or "Внутриклеточный кальций")
    _apply_style(ax)
    return fig, ax


def plot_force(
    result: SimulationResult,
    component: str = "F_XSE",
    *,
    last_n_ms: float | None = None,
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Механическая сила F(t).

    Аргумент component: 'F_XSE' | 'F_CE' | 'F_SE' | 'F_PE' | 'F_VS1' | 'F_VS2'
    """
    if component not in result.forces:
        raise ValueError(
            f"Нет компоненты '{component}'. Доступно: {list(result.forces)}"
        )
    fig, ax = _get_ax(ax)
    t, f = _slice_last(result.time, result.forces[component], last_n_ms)
    ax.plot(t, f, color="#1a9641", linewidth=0.8)
    ax.set_xlabel("Время, мс")
    ax.set_ylabel(f"{component}, мН")
    ax.set_title(title or f"Сила {component}")
    _apply_style(ax)
    return fig, ax


def plot_currents(
    result: SimulationResult,
    currents: list[str] | None = None,
    *,
    last_n_ms: float | None = None,
    title: str | None = None,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Набор ионных токов на отдельных подграфиках.

    Аргумент currents — список имён токов. По умолчанию: основные 6.
    """
    if currents is None:
        currents = ["i_Na", "i_CaL", "i_NaCa", "i_NaK", "i_K1", "i_Kr"]

    n = len(currents)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.2 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, currents):
        if name not in result.currents:
            ax.set_visible(False)
            continue
        t, cur = _slice_last(result.time, result.currents[name], last_n_ms)
        ax.plot(t, cur, linewidth=0.8)
        ax.set_ylabel(f"{name}\nпА/пФ", fontsize=8)
        _apply_style(ax)

    axes[-1].set_xlabel("Время, мс")
    fig.suptitle(title or "Ионные токи", y=1.01)
    fig.tight_layout()
    return fig, axes


def plot_sr_calcium(
    result: SimulationResult,
    *,
    last_n_ms: float | None = None,
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Ca_nSR и Ca_jSR на одном графике."""
    fig, ax = _get_ax(ax)
    t_nsr, ca_nsr = _slice_last(result.time, result.variables["Ca_nSR"], last_n_ms)
    _, ca_jsr = _slice_last(result.time, result.variables["Ca_jSR"], last_n_ms)
    ax.plot(t_nsr, ca_nsr, label="Ca_nSR", color="#4575b4", linewidth=0.8)
    ax.plot(t_nsr, ca_jsr, label="Ca_jSR", color="#d73027", linewidth=0.8)
    ax.set_xlabel("Время, мс")
    ax.set_ylabel("[Ca²⁺]SR, мМ")
    ax.set_title(title or "Кальций в SR")
    ax.legend(fontsize=8)
    _apply_style(ax)
    return fig, ax


def plot_overview(
    result: SimulationResult,
    *,
    last_n_ms: float | None = None,
    title: str | None = None,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Сводный график: V, Ca_i, F_XSE на трёх подграфиках.

    Типичное использование для быстрой проверки результата прогона.
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    plot_voltage(result, last_n_ms=last_n_ms, ax=axes[0])
    plot_calcium(result, last_n_ms=last_n_ms, ax=axes[1])
    plot_force(result, last_n_ms=last_n_ms, ax=axes[2])

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig, axes.tolist()


# ---------------------------------------------------------------------------
# Параметрический sweep
# ---------------------------------------------------------------------------


def plot_sweep(
    results: Sequence[SimulationResult],
    variable: str,
    sweep_param: str,
    *,
    last_n_ms: float | None = None,
    from_forces: bool = False,
    from_currents: bool = False,
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Наложить кривые нескольких прогонов на один график.

    Аргументы:
        results      — список результатов из run_sweep()
        variable     — имя переменной ('V', 'Ca_i', 'F_XSE', 'i_Na', ...)
        sweep_param  — имя параметра из TNNPMParams для подписи в легенде
        from_forces  — искать переменную в result.forces
        from_currents — искать переменную в result.currents
                        (по умолчанию ищем в result.variables)

    Пример:
        fig, ax = plot_sweep(results, 'Ca_i', 'g_CaL')
    """
    fig, ax = _get_ax(ax)

    for i, r in enumerate(results):
        color = _COLORS[i % len(_COLORS)]
        param_val = getattr(r.params, sweep_param, "?")
        label = f"{sweep_param}={param_val:.3g}"

        if from_forces:
            arr = r.forces[variable]
        elif from_currents:
            arr = r.currents[variable]
        else:
            arr = r.variables[variable]

        t, y = _slice_last(r.time, arr, last_n_ms)
        ax.plot(t, y, color=color, linewidth=0.8, label=label)

    ax.set_xlabel("Время, мс")
    ax.set_ylabel(variable)
    ax.set_title(title or f"Sweep по {sweep_param}: {variable}")
    ax.legend(fontsize=8, loc="upper right")
    _apply_style(ax)
    return fig, ax


def plot_sweep_peak(
    results: Sequence[SimulationResult],
    variable: str,
    sweep_param: str,
    *,
    last_n_ms: float | None = None,
    from_forces: bool = False,
    from_currents: bool = False,
    stat: str = "max",
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """График «пиковое значение vs параметр» для sweep.

    Аргумент stat: 'max' | 'min' | 'mean' | 'last'

    Пример:
        fig, ax = plot_sweep_peak(results, 'F_XSE', 'g_CaL',
                                  from_forces=True, stat='max')
    """
    stat_fn = {"max": np.max, "min": np.min, "mean": np.mean, "last": lambda x: x[-1]}
    if stat not in stat_fn:
        raise ValueError(f"stat должен быть одним из {list(stat_fn)}")

    param_vals, stat_vals = [], []
    for r in results:
        if from_forces:
            arr = r.forces[variable]
        elif from_currents:
            arr = r.currents[variable]
        else:
            arr = r.variables[variable]

        _, y = _slice_last(r.time, arr, last_n_ms)
        param_vals.append(getattr(r.params, sweep_param))
        stat_vals.append(stat_fn[stat](y))

    fig, ax = _get_ax(ax)
    ax.plot(param_vals, stat_vals, "o-", color="#2c7bb6", linewidth=1.2, markersize=5)
    ax.set_xlabel(sweep_param)
    ax.set_ylabel(f"{stat}({variable})")
    ax.set_title(title or f"{stat}({variable}) vs {sweep_param}")
    _apply_style(ax)
    return fig, ax


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------


def _get_ax(ax: plt.Axes | None) -> tuple[plt.Figure, plt.Axes]:
    """Создать новую Figure или использовать переданную Axes."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3.5))
    else:
        fig = ax.get_figure()
    return fig, ax


def _slice_last(
    t: np.ndarray,
    y: np.ndarray,
    last_n_ms: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Вернуть последние last_n_ms миллисекунд трассы."""
    if last_n_ms is None:
        return t, y
    mask = t >= (t[-1] - last_n_ms)
    return t[mask], y[mask]


def _apply_style(ax: plt.Axes) -> None:
    """Минималистичное оформление осей."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3g"))
