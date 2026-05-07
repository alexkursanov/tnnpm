"""
test_experiment.py
==================
Тесты для experiment.py.

Солвер мокируется — тесты проверяют логику experiment.py,
а не корректность численного интегрирования.

Запуск:
    pytest test_experiment.py -v
"""

from __future__ import annotations

import itertools
from unittest.mock import patch

import numpy as np
import pytest

from experiment import (
    SimulationResult,
    _VAR_NAMES,
    _make_rhs_fn,
    _y_to_variables,
    run_single,
    run_sweep,
    sweep_combinations,
)
from model import make_init_state, rhs, _N_STATES
from parameters import DEFAULT_PARAMS, InitialState, SimConfig, TNNPMParams

# ---------------------------------------------------------------------------
# Фикстуры
# ---------------------------------------------------------------------------

N_PTS = 10  # число точек в мок-результате


@pytest.fixture
def p():
    return TNNPMParams()


@pytest.fixture
def state_and_l0(p):
    return make_init_state(p)


@pytest.fixture
def state0(state_and_l0):
    return state_and_l0[0]


@pytest.fixture
def l0(state_and_l0):
    return state_and_l0[1]


@pytest.fixture
def config():
    return SimConfig(time_stop=100.0, n_out=N_PTS)


@pytest.fixture
def mock_t_y(state0):
    """Мок-результат солвера: N_PTS точек, постоянное состояние."""
    t = np.linspace(0.0, 100.0, N_PTS)
    y0 = np.array(state0.to_array())
    y = np.tile(y0, (N_PTS, 1))
    return t, y


@pytest.fixture
def mock_integrate(mock_t_y):
    """Патч integrate() чтобы не запускать реальный CVode."""
    t, y = mock_t_y
    with patch("experiment.integrate", return_value=(t, y)) as m:
        yield m


# ---------------------------------------------------------------------------
# 1. SimulationResult
# ---------------------------------------------------------------------------


class TestSimulationResult:

    def test_frozen(self, p, state0, config, mock_integrate):
        r = run_single(p, state0, config, l0=0.44)
        with pytest.raises((TypeError, Exception)):
            r.time = np.zeros(1)

    def test_V_property(self, p, state0, config, mock_integrate):
        r = run_single(p, state0, config, l0=0.44)
        np.testing.assert_array_equal(r.V, r.variables["V"])

    def test_Ca_i_property(self, p, state0, config, mock_integrate):
        r = run_single(p, state0, config, l0=0.44)
        np.testing.assert_array_equal(r.Ca_i, r.variables["Ca_i"])

    def test_F_XSE_property(self, p, state0, config, mock_integrate):
        r = run_single(p, state0, config, l0=0.44)
        np.testing.assert_array_equal(r.F_XSE, r.forces["F_XSE"])

    def test_last_state_returns_InitialState(self, p, state0, config, mock_integrate):
        r = run_single(p, state0, config, l0=0.44)
        s = r.last_state()
        assert isinstance(s, InitialState)

    def test_last_state_V_matches(self, p, state0, config, mock_integrate):
        r = run_single(p, state0, config, l0=0.44)
        s = r.last_state()
        assert s.V == pytest.approx(r.variables["V"][-1])

    def test_meta_contains_wall_time(self, p, state0, config, mock_integrate):
        r = run_single(p, state0, config, l0=0.44)
        assert "wall_time_s" in r.meta
        assert r.meta["wall_time_s"] >= 0.0

    def test_meta_n_points(self, p, state0, config, mock_integrate):
        r = run_single(p, state0, config, l0=0.44)
        assert r.meta["n_points"] == N_PTS

    def test_params_stored(self, p, state0, config, mock_integrate):
        r = run_single(p, state0, config, l0=0.44)
        assert r.params is p

    def test_config_stored(self, p, state0, config, mock_integrate):
        r = run_single(p, state0, config, l0=0.44)
        assert r.config is config


# ---------------------------------------------------------------------------
# 2. _y_to_variables
# ---------------------------------------------------------------------------


class TestYToVariables:

    def test_all_var_names_present(self, state0):
        y0 = np.array(state0.to_array())
        y = np.tile(y0, (5, 1))
        vs = _y_to_variables(y)
        for name in _VAR_NAMES:
            assert name in vs, f"Переменная '{name}' отсутствует"

    def test_shape(self, state0):
        N = 7
        y0 = np.array(state0.to_array())
        y = np.tile(y0, (N, 1))
        vs = _y_to_variables(y)
        for name, arr in vs.items():
            assert arr.shape == (N,), f"'{name}' shape = {arr.shape}"

    def test_V_value(self, state0):
        y0 = np.array(state0.to_array())
        y = np.tile(y0, (3, 1))
        vs = _y_to_variables(y)
        assert vs["V"][0] == pytest.approx(state0.V)

    def test_n_variables(self):
        assert len(_VAR_NAMES) == _N_STATES


# ---------------------------------------------------------------------------
# 3. _make_rhs_fn
# ---------------------------------------------------------------------------


class TestMakeRhsFn:

    def test_returns_callable(self, p, l0):
        fn = _make_rhs_fn(p, 0.0, l0)
        assert callable(fn)

    def test_output_shape(self, p, state0, l0):
        fn = _make_rhs_fn(p, 0.0, l0)
        y0 = np.array(state0.to_array(), dtype=np.float64)
        dy = fn(0.0, y0)
        assert dy.shape == (33,)

    def test_consistent_with_rhs(self, p, state0, l0):
        """Замыкание даёт тот же результат что rhs() напрямую."""
        fn = _make_rhs_fn(p, 0.0, l0)
        y0 = np.array(state0.to_array(), dtype=np.float64)
        dy_fn = fn(0.0, y0)
        dy_rhs = rhs(0.0, y0, p, F_afterload=0.0, l0=l0)
        np.testing.assert_array_equal(dy_fn, dy_rhs)

    def test_different_params_different_result(self, p, state0, l0):
        p2 = p.replace(g_Na=0.0)
        y0 = np.array(state0.to_array(), dtype=np.float64)
        dy1 = _make_rhs_fn(p, 0.0, l0)(0.0, y0)
        dy2 = _make_rhs_fn(p2, 0.0, l0)(0.0, y0)
        assert not np.allclose(dy1, dy2)


# ---------------------------------------------------------------------------
# 4. run_single
# ---------------------------------------------------------------------------


class TestRunSingle:

    def test_returns_SimulationResult(self, p, state0, config, mock_integrate):
        r = run_single(p, state0, config, l0=0.44)
        assert isinstance(r, SimulationResult)

    def test_time_shape(self, p, state0, config, mock_integrate):
        r = run_single(p, state0, config, l0=0.44)
        assert r.time.shape == (N_PTS,)

    def test_variables_shape(self, p, state0, config, mock_integrate):
        r = run_single(p, state0, config, l0=0.44)
        for name, arr in r.variables.items():
            assert arr.shape == (N_PTS,), f"'{name}' shape = {arr.shape}"

    def test_currents_keys(self, p, state0, config, mock_integrate):
        r = run_single(p, state0, config, l0=0.44)
        expected = [
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
        for k in expected:
            assert k in r.currents, f"'{k}' отсутствует в currents"

    def test_forces_keys(self, p, state0, config, mock_integrate):
        r = run_single(p, state0, config, l0=0.44)
        for k in ["F_CE", "F_SE", "F_PE", "F_VS1", "F_VS2", "F_XSE"]:
            assert k in r.forces, f"'{k}' отсутствует в forces"

    def test_state0_none_calls_make_init_state(self, p, config):
        """Если state0=None, make_init_state вызывается автоматически."""
        with patch(
            "experiment.make_init_state", wraps=make_init_state
        ) as mock_mis, patch("experiment.integrate") as mock_int:
            s, l0_val = make_init_state(p)
            mock_int.return_value = (
                np.linspace(0, 100, N_PTS),
                np.tile(np.array(s.to_array()), (N_PTS, 1)),
            )
            run_single(p, state0=None, config=config)
            mock_mis.assert_called_once()

    def test_integrate_called_once(self, p, state0, config, mock_integrate):
        run_single(p, state0, config, l0=0.44)
        mock_integrate.assert_called_once()

    def test_passed_params_used_in_rhs(self, p, state0, config, mock_t_y):
        """rhs вызывается с теми параметрами что переданы в run_single."""
        t, y = mock_t_y
        captured = {}

        def fake_integrate(rhs_fn, y0, cfg):
            # Вызываем rhs_fn один раз и запоминаем результат
            captured["dy"] = rhs_fn(0.0, y0)
            return t, y

        p_zero_gNa = p.replace(g_Na=0.0)
        with patch("experiment.integrate", side_effect=fake_integrate):
            run_single(p_zero_gNa, state0, config, l0=0.44)

        # Проверяем через прямой вызов rhs с теми же параметрами
        y0 = np.array(state0.to_array(), dtype=np.float64)
        expected = rhs(0.0, y0, p_zero_gNa, F_afterload=0.0, l0=0.44)
        np.testing.assert_allclose(captured["dy"], expected)

    def test_chain_runs(self, p, config, mock_t_y):
        """last_state() можно передать как state0 следующего прогона."""
        t, y = mock_t_y
        with patch("experiment.integrate", return_value=(t, y)):
            s0, l0_val = make_init_state(p)
            r1 = run_single(p, s0, config, l0=l0_val)
            r2 = run_single(p, r1.last_state(), config, l0=l0_val)
        assert isinstance(r2, SimulationResult)

    def test_recompute_init_mechanics(self, p, state0, config, mock_t_y):
        """recompute_init_mechanics=True вызывает make_init_state."""
        t, y = mock_t_y
        with patch(
            "experiment.make_init_state", wraps=make_init_state
        ) as mock_mis, patch("experiment.integrate", return_value=(t, y)):
            run_single(p, state0, config, recompute_init_mechanics=True)
            mock_mis.assert_called_once_with(p, base=state0)

    def test_f_afterload_stored_in_meta(self, p, state0, config, mock_integrate):
        r = run_single(p, state0, config, l0=0.44, F_afterload=1.5)
        assert r.meta["F_afterload"] == 1.5


# ---------------------------------------------------------------------------
# 5. run_sweep
# ---------------------------------------------------------------------------


class TestRunSweep:

    def test_returns_list(self, p, state0, config, mock_t_y):
        t, y = mock_t_y
        with patch("experiment.integrate", return_value=(t, y)):
            results = run_sweep(
                sweep={"g_Na": [14.0, 16.0]},
                base_params=p,
                state0=state0,
                config=config,
                l0=0.44,
            )
        assert isinstance(results, list)

    def test_correct_number_of_results_1d(self, p, state0, config, mock_t_y):
        t, y = mock_t_y
        with patch("experiment.integrate", return_value=(t, y)):
            results = run_sweep(
                sweep={"g_Na": [14.0, 15.0, 16.0]},
                base_params=p,
                state0=state0,
                config=config,
                l0=0.44,
            )
        assert len(results) == 3

    def test_correct_number_of_results_2d(self, p, state0, config, mock_t_y):
        """Декартово произведение: 2 × 3 = 6 прогонов."""
        t, y = mock_t_y
        with patch("experiment.integrate", return_value=(t, y)):
            results = run_sweep(
                sweep={"g_Na": [14.0, 16.0], "g_CaL": [3e-5, 5e-5, 7e-5]},
                base_params=p,
                state0=state0,
                config=config,
                l0=0.44,
            )
        assert len(results) == 6

    def test_each_result_has_correct_param(self, p, state0, config, mock_t_y):
        """Каждый результат содержит параметры из своей комбинации."""
        t, y = mock_t_y
        g_Na_values = [12.0, 14.0, 16.0]
        with patch("experiment.integrate", return_value=(t, y)):
            results = run_sweep(
                sweep={"g_Na": g_Na_values},
                base_params=p,
                state0=state0,
                config=config,
                l0=0.44,
            )
        for r, g_Na in zip(results, g_Na_values):
            assert r.params.g_Na == g_Na

    def test_base_params_not_mutated(self, p, state0, config, mock_t_y):
        """base_params не изменяются в процессе sweep."""
        t, y = mock_t_y
        g_Na_orig = p.g_Na
        with patch("experiment.integrate", return_value=(t, y)):
            run_sweep(
                sweep={"g_Na": [0.0, 1.0, 2.0]},
                base_params=p,
                state0=state0,
                config=config,
                l0=0.44,
            )
        assert p.g_Na == g_Na_orig

    def test_order_matches_product(self, p, state0, config, mock_t_y):
        """Порядок результатов совпадает с itertools.product."""
        t, y = mock_t_y
        g_vals = [10.0, 20.0]
        ca_vals = [3e-5, 5e-5]
        with patch("experiment.integrate", return_value=(t, y)):
            results = run_sweep(
                sweep={"g_Na": g_vals, "g_CaL": ca_vals},
                base_params=p,
                state0=state0,
                config=config,
                l0=0.44,
            )
        expected = list(itertools.product(g_vals, ca_vals))
        for r, (g_Na, g_CaL) in zip(results, expected):
            assert r.params.g_Na == g_Na
            assert r.params.g_CaL == pytest.approx(g_CaL)

    def test_empty_sweep_returns_one_result(self, p, state0, config, mock_t_y):
        """Пустой sweep (все списки из одного значения) → 1 прогон."""
        t, y = mock_t_y
        with patch("experiment.integrate", return_value=(t, y)):
            results = run_sweep(
                sweep={"g_Na": [14.838]},
                base_params=p,
                state0=state0,
                config=config,
                l0=0.44,
            )
        assert len(results) == 1

    def test_all_results_are_SimulationResult(self, p, state0, config, mock_t_y):
        t, y = mock_t_y
        with patch("experiment.integrate", return_value=(t, y)):
            results = run_sweep(
                sweep={"g_Na": [14.0, 16.0]},
                base_params=p,
                state0=state0,
                config=config,
                l0=0.44,
            )
        for r in results:
            assert isinstance(r, SimulationResult)


# ---------------------------------------------------------------------------
# 6. sweep_combinations
# ---------------------------------------------------------------------------


class TestSweepCombinations:

    def test_1d(self):
        combos = sweep_combinations({"g_Na": [10.0, 14.0, 18.0]})
        assert len(combos) == 3
        assert combos[0] == {"g_Na": 10.0}
        assert combos[2] == {"g_Na": 18.0}

    def test_2d(self):
        combos = sweep_combinations({"a": [1, 2], "b": [10, 20, 30]})
        assert len(combos) == 6

    def test_keys_present(self):
        combos = sweep_combinations({"g_Na": [1.0], "g_CaL": [2.0]})
        assert "g_Na" in combos[0]
        assert "g_CaL" in combos[0]

    def test_order(self):
        combos = sweep_combinations({"x": [1, 2], "y": [10, 20]})
        expected = [
            {"x": 1, "y": 10},
            {"x": 1, "y": 20},
            {"x": 2, "y": 10},
            {"x": 2, "y": 20},
        ]
        assert combos == expected

    def test_returns_list_of_dicts(self):
        combos = sweep_combinations({"a": [1, 2]})
        assert all(isinstance(c, dict) for c in combos)
