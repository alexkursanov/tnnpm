"""
test_io_plots.py
================
Тесты для io.py и plots.py.

io   — тестируется с реальным HDF5 во временной директории (tmp_path).
plots — тестируется без отображения окон (matplotlib backend='Agg').

Запуск:
    pytest test_io_plots.py -v
"""

from __future__ import annotations

import dataclasses
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")  # без GUI — до импорта pyplot
import matplotlib.pyplot as plt
import numpy as np
import pytest

from experiment import SimulationResult
from simulation_io import (
    load,
    load_batch,
    load_legacy,
    load_state,
    save,
    save_batch,
    _read_group_recursive,
)
from model import make_init_state, _N_STATES
from parameters import DEFAULT_PARAMS, InitialState, SimConfig, TNNPMParams
from plots import (
    _slice_last,
    plot_calcium,
    plot_force,
    plot_overview,
    plot_sr_calcium,
    plot_sweep,
    plot_sweep_peak,
    plot_voltage,
)

# ---------------------------------------------------------------------------
# Фикстуры
# ---------------------------------------------------------------------------

N_PTS = 20


@pytest.fixture
def p():
    return TNNPMParams()


@pytest.fixture
def state0(p):
    s, _ = make_init_state(p)
    return s


@pytest.fixture
def l0(p):
    _, l0 = make_init_state(p)
    return l0


@pytest.fixture
def dummy_result(p, state0, l0):
    """SimulationResult с синтетическими данными."""
    t = np.linspace(0.0, 1000.0, N_PTS)
    y0 = np.array(state0.to_array())
    y = np.tile(y0, (N_PTS, 1))

    from experiment import _y_to_variables, calculate_outputs

    variables = _y_to_variables(y)
    outputs = calculate_outputs(t, y, p, l0=l0)

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
    config = SimConfig(time_stop=1000.0, n_out=N_PTS)

    return SimulationResult(
        time=t,
        variables=variables,
        currents=currents,
        forces=forces,
        params=p,
        state0=state0,
        config=config,
        meta={"wall_time_s": 1.23, "n_points": N_PTS, "F_afterload": 0.0, "l0": l0},
    )


@pytest.fixture
def sweep_results(dummy_result):
    """Три результата с разными g_CaL для тестов sweep."""
    scales = [0.8, 1.0, 1.2]
    results = []
    for s in scales:
        p2 = dummy_result.params.replace(g_CaL=dummy_result.params.g_CaL * s)
        r = dataclasses.replace(dummy_result, params=p2)
        results.append(r)
    return results


# ===========================================================================
# 1. io — save / load roundtrip
# ===========================================================================


class TestSaveLoad:

    def test_roundtrip_time(self, dummy_result, tmp_path):
        path = tmp_path / "result.h5"
        save(dummy_result, path)
        r = load(path)
        np.testing.assert_allclose(r.time, dummy_result.time)

    def test_roundtrip_variables(self, dummy_result, tmp_path):
        path = tmp_path / "result.h5"
        save(dummy_result, path)
        r = load(path)
        for name in dummy_result.variables:
            np.testing.assert_allclose(
                r.variables[name],
                dummy_result.variables[name],
                err_msg=f"Переменная '{name}' не совпадает после roundtrip",
            )

    def test_roundtrip_currents(self, dummy_result, tmp_path):
        path = tmp_path / "result.h5"
        save(dummy_result, path)
        r = load(path)
        for name in dummy_result.currents:
            np.testing.assert_allclose(r.currents[name], dummy_result.currents[name])

    def test_roundtrip_forces(self, dummy_result, tmp_path):
        path = tmp_path / "result.h5"
        save(dummy_result, path)
        r = load(path)
        for name in dummy_result.forces:
            np.testing.assert_allclose(r.forces[name], dummy_result.forces[name])

    def test_roundtrip_params(self, dummy_result, tmp_path):
        path = tmp_path / "result.h5"
        save(dummy_result, path)
        r = load(path)
        # Проверяем несколько ключевых параметров
        assert r.params.g_Na == pytest.approx(dummy_result.params.g_Na)
        assert r.params.g_CaL == pytest.approx(dummy_result.params.g_CaL)
        assert r.params.stim_period == pytest.approx(dummy_result.params.stim_period)

    def test_roundtrip_state0_V(self, dummy_result, tmp_path):
        path = tmp_path / "result.h5"
        save(dummy_result, path)
        r = load(path)
        assert r.state0.V == pytest.approx(dummy_result.state0.V)

    def test_roundtrip_config(self, dummy_result, tmp_path):
        path = tmp_path / "result.h5"
        save(dummy_result, path)
        r = load(path)
        assert r.config.time_stop == pytest.approx(dummy_result.config.time_stop)
        assert r.config.atol == pytest.approx(dummy_result.config.atol)

    def test_file_created(self, dummy_result, tmp_path):
        path = tmp_path / "result.h5"
        save(dummy_result, path)
        assert path.exists()

    def test_parent_dir_created(self, dummy_result, tmp_path):
        path = tmp_path / "nested" / "dir" / "result.h5"
        save(dummy_result, path)
        assert path.exists()

    def test_load_returns_SimulationResult(self, dummy_result, tmp_path):
        path = tmp_path / "result.h5"
        save(dummy_result, path)
        r = load(path)
        assert isinstance(r, SimulationResult)


# ===========================================================================
# 2. io — save_batch / load_batch
# ===========================================================================


class TestBatch:

    def test_batch_roundtrip_count(self, sweep_results, tmp_path):
        path = tmp_path / "sweep.h5"
        save_batch(sweep_results, path)
        loaded = load_batch(path)
        assert len(loaded) == len(sweep_results)

    def test_batch_roundtrip_params(self, sweep_results, tmp_path):
        path = tmp_path / "sweep.h5"
        save_batch(sweep_results, path)
        loaded = load_batch(path)
        for orig, loaded_r in zip(sweep_results, loaded):
            assert loaded_r.params.g_CaL == pytest.approx(orig.params.g_CaL)

    def test_batch_roundtrip_time(self, sweep_results, tmp_path):
        path = tmp_path / "sweep.h5"
        save_batch(sweep_results, path)
        loaded = load_batch(path)
        for orig, lr in zip(sweep_results, loaded):
            np.testing.assert_allclose(lr.time, orig.time)

    def test_batch_empty(self, tmp_path):
        path = tmp_path / "empty.h5"
        save_batch([], path)
        loaded = load_batch(path)
        assert loaded == []


# ===========================================================================
# 3. io — load_state
# ===========================================================================


class TestLoadState:

    def test_returns_InitialState(self, dummy_result, tmp_path):
        path = tmp_path / "result.h5"
        save(dummy_result, path)
        s = load_state(path)
        assert isinstance(s, InitialState)

    def test_V_matches_last_point(self, dummy_result, tmp_path):
        path = tmp_path / "result.h5"
        save(dummy_result, path)
        s = load_state(path)
        assert s.V == pytest.approx(dummy_result.variables["V"][-1])

    def test_Ca_i_matches_last_point(self, dummy_result, tmp_path):
        path = tmp_path / "result.h5"
        save(dummy_result, path)
        s = load_state(path)
        assert s.Ca_i == pytest.approx(dummy_result.variables["Ca_i"][-1])


# ===========================================================================
# 4. io — load_legacy
# ===========================================================================


class TestLoadLegacy:

    def test_returns_dict(self, tmp_path):
        """Файл в старом формате (h5py напрямую) читается как dict."""
        import h5py

        path = tmp_path / "legacy.h5"
        with h5py.File(path, "w") as f:
            grp = f.create_group("variables")
            grp.create_dataset("V", data=np.array([-85.0, -84.0]))

        d = load_legacy(path)
        assert isinstance(d, dict)
        assert "variables" in d
        np.testing.assert_allclose(d["variables"]["V"], [-85.0, -84.0])

    def test_nested_groups(self, tmp_path):
        import h5py

        path = tmp_path / "legacy_nested.h5"
        with h5py.File(path, "w") as f:
            f.create_group("a").create_group("b").create_dataset(
                "x", data=np.array([1.0])
            )

        d = load_legacy(path)
        assert d["a"]["b"]["x"][0] == pytest.approx(1.0)


# ===========================================================================
# 5. plots — базовые свойства
# ===========================================================================


class TestPlotsBasic:

    def test_plot_voltage_returns_fig_ax(self, dummy_result):
        fig, ax = plot_voltage(dummy_result)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_plot_calcium_returns_fig_ax(self, dummy_result):
        fig, ax = plot_calcium(dummy_result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_force_returns_fig_ax(self, dummy_result):
        fig, ax = plot_force(dummy_result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_sr_calcium_returns_fig_ax(self, dummy_result):
        fig, ax = plot_sr_calcium(dummy_result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_overview_returns_three_axes(self, dummy_result):
        fig, axes = plot_overview(dummy_result)
        assert len(axes) == 3
        plt.close(fig)

    def test_plot_force_invalid_component(self, dummy_result):
        with pytest.raises(ValueError, match="Нет компоненты"):
            plot_force(dummy_result, component="F_NONEXISTENT")

    def test_each_call_creates_new_figure(self, dummy_result):
        """Повторные вызовы создают независимые Figure — нет глобального состояния."""
        fig1, _ = plot_voltage(dummy_result)
        fig2, _ = plot_voltage(dummy_result)
        assert fig1 is not fig2
        plt.close(fig1)
        plt.close(fig2)

    def test_plot_reuses_passed_ax(self, dummy_result):
        """Если передана ax, возвращается та же Figure."""
        fig_ext, ax_ext = plt.subplots()
        fig_ret, ax_ret = plot_voltage(dummy_result, ax=ax_ext)
        assert fig_ret is fig_ext
        assert ax_ret is ax_ext
        plt.close(fig_ext)

    def test_plot_voltage_has_data(self, dummy_result):
        """На графике есть линия с данными."""
        fig, ax = plot_voltage(dummy_result)
        lines = ax.get_lines()
        assert len(lines) >= 1
        assert len(lines[0].get_xdata()) == N_PTS
        plt.close(fig)

    def test_plot_force_all_components(self, dummy_result):
        for comp in ["F_CE", "F_SE", "F_PE", "F_VS1", "F_VS2", "F_XSE"]:
            fig, ax = plot_force(dummy_result, component=comp)
            plt.close(fig)


# ===========================================================================
# 6. plots — _slice_last
# ===========================================================================


class TestSliceLast:

    def test_none_returns_all(self):
        t = np.linspace(0, 100, 50)
        y = np.ones(50)
        t2, y2 = _slice_last(t, y, None)
        np.testing.assert_array_equal(t2, t)
        np.testing.assert_array_equal(y2, y)

    def test_slice_correct_length(self):
        t = np.linspace(0, 1000, 1000)
        y = np.ones(1000)
        t2, y2 = _slice_last(t, y, last_n_ms=100.0)
        assert t2[0] >= 900.0
        assert t2[-1] == pytest.approx(1000.0)

    def test_slice_preserves_values(self):
        t = np.array([0.0, 500.0, 900.0, 1000.0])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        t2, y2 = _slice_last(t, y, last_n_ms=200.0)
        np.testing.assert_array_equal(t2, [900.0, 1000.0])
        np.testing.assert_array_equal(y2, [3.0, 4.0])

    def test_last_n_ms_larger_than_trace(self):
        """Если last_n_ms > длины трассы — возвращается вся трасса."""
        t = np.linspace(0, 100, 50)
        y = np.ones(50)
        t2, y2 = _slice_last(t, y, last_n_ms=9999.0)
        assert len(t2) == 50


# ===========================================================================
# 7. plots — sweep
# ===========================================================================


class TestPlotSweep:

    def test_plot_sweep_returns_fig_ax(self, sweep_results):
        fig, ax = plot_sweep(sweep_results, "V", "g_CaL")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_sweep_n_lines(self, sweep_results):
        """На графике столько линий, сколько результатов."""
        fig, ax = plot_sweep(sweep_results, "V", "g_CaL")
        assert len(ax.get_lines()) == len(sweep_results)
        plt.close(fig)

    def test_plot_sweep_legend_labels(self, sweep_results):
        """Легенда содержит значения параметра sweep."""
        fig, ax = plot_sweep(sweep_results, "V", "g_CaL")
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert all("g_CaL" in lbl for lbl in labels)
        plt.close(fig)

    def test_plot_sweep_from_forces(self, sweep_results):
        fig, ax = plot_sweep(sweep_results, "F_XSE", "g_CaL", from_forces=True)
        assert len(ax.get_lines()) == len(sweep_results)
        plt.close(fig)

    def test_plot_sweep_from_currents(self, sweep_results):
        fig, ax = plot_sweep(sweep_results, "i_Na", "g_CaL", from_currents=True)
        assert len(ax.get_lines()) == len(sweep_results)
        plt.close(fig)

    def test_plot_sweep_peak_returns_fig_ax(self, sweep_results):
        fig, ax = plot_sweep_peak(sweep_results, "Ca_i", "g_CaL")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_sweep_peak_n_points(self, sweep_results):
        """Столько точек на графике, сколько результатов."""
        fig, ax = plot_sweep_peak(sweep_results, "Ca_i", "g_CaL")
        lines = ax.get_lines()
        assert len(lines[0].get_xdata()) == len(sweep_results)
        plt.close(fig)

    def test_plot_sweep_peak_invalid_stat(self, sweep_results):
        with pytest.raises(ValueError, match="stat должен быть"):
            plot_sweep_peak(sweep_results, "Ca_i", "g_CaL", stat="median")

    @pytest.mark.parametrize("stat", ["max", "min", "mean", "last"])
    def test_plot_sweep_peak_all_stats(self, sweep_results, stat):
        fig, ax = plot_sweep_peak(sweep_results, "Ca_i", "g_CaL", stat=stat)
        plt.close(fig)
