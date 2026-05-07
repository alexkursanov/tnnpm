"""
test_model.py
=============
Тесты для parameters.py и model.py.

Категории:
  1. TNNPMParams / InitialState / SimConfig — dataclass API
  2. InitialState — сериализация и загрузка
  3. make_init_state — механическое равновесие
  4. rhs() — физические инварианты
  5. rhs() — численные свойства
  6. Вспомогательные функции механики
  7. _gate_CaL — зависимость fCass от Ca_ss
  8. calculate_outputs — соответствие rhs()
  9. Стимул — периодичность

Запуск:
    pytest test_model.py -v
"""

import dataclasses

import numpy as np
import pytest

from model import (
    _L,
    _N0,
    _compute_calcium,
    _compute_electrical,
    _compute_mechanical,
    _fi,
    _gate_CaL,
    _gate_Kr,
    _gate_Na,
    _gate_Ks,
    _gate_to,
    _p_v,
    _p_prime_v,
    _pi_N_A,
    _stim_current,
    calculate_outputs,
    make_init_state,
    rhs,
    _N_STATES,
    _I_V,
    _I_CA_I,
    _I_CA_SS,
    _I_NA_I,
    _I_K_I,
    _I_CA_NSR,
    _I_CA_JSR,
    _I_N,
    _I_A,
    _I_L1,
    _I_L2,
    _I_L3,
    _I_V_MECH,
    _I_W_MECH,
    _I_FCASS,
)
from parameters import (
    DEFAULT_PARAMS,
    DEFAULT_STATE,
    InitialState,
    SimConfig,
    TNNPMParams,
)

# ---------------------------------------------------------------------------
# Фикстуры
# ---------------------------------------------------------------------------


@pytest.fixture
def p():
    return TNNPMParams()


@pytest.fixture
def state0_and_l0(p):
    return make_init_state(p)


@pytest.fixture
def y0(state0_and_l0):
    state0, _ = state0_and_l0
    return np.array(state0.to_array(), dtype=np.float64)


@pytest.fixture
def l0(state0_and_l0):
    _, l0 = state0_and_l0
    return l0


# ===========================================================================
# 1. TNNPMParams / InitialState / SimConfig — dataclass API
# ===========================================================================


class TestDataclassAPI:

    def test_params_frozen(self, p):
        """Параметры нельзя изменить напрямую."""
        with pytest.raises((dataclasses.FrozenInstanceError, TypeError)):
            p.g_Na = 0.0

    def test_state_frozen(self):
        s = InitialState()
        with pytest.raises((dataclasses.FrozenInstanceError, TypeError)):
            s.V = 0.0

    def test_params_replace(self, p):
        """replace() создаёт новый объект с изменёнными полями."""
        p2 = p.replace(g_Na=99.0)
        assert p2.g_Na == 99.0
        assert p.g_Na == 14.838  # оригинал не изменился
        assert p2 is not p

    def test_state_replace(self):
        s = InitialState()
        s2 = s.replace(V=-70.0)
        assert s2.V == -70.0
        assert s.V == pytest.approx(-85.927, rel=1e-3)

    def test_params_to_dict_roundtrip(self, p):
        d = p.to_dict()
        p2 = TNNPMParams.from_dict(d)
        assert p == p2

    def test_params_from_dict_ignores_unknown_keys(self, p):
        """from_dict() не падает на устаревших ключах из старых файлов."""
        d = p.to_dict()
        d["obsolete_key_xyz"] = 999.0
        p2 = TNNPMParams.from_dict(d)
        assert p == p2

    def test_simconfig_defaults(self):
        cfg = SimConfig()
        assert cfg.time_start == 0.0
        assert cfg.atol == 1e-9

    def test_default_params_singleton(self):
        """DEFAULT_PARAMS — это DEFAULT_PARAMS, не копия."""
        assert DEFAULT_PARAMS is DEFAULT_PARAMS
        assert DEFAULT_PARAMS == TNNPMParams()


# ===========================================================================
# 2. InitialState — сериализация и загрузка
# ===========================================================================


class TestInitialState:

    def test_to_array_length(self):
        y = InitialState().to_array()
        assert len(y) == _N_STATES

    def test_array_roundtrip(self):
        s = InitialState()
        y = s.to_array()
        s2 = InitialState.from_array(y)
        assert s == s2

    def test_from_result(self):
        """from_result() загружает последнюю точку из словаря массивов."""
        s = InitialState()
        # Строим фиктивный variables-словарь: каждое поле — массив из двух точек
        variables = {
            "d": np.array([0.0, s.d]),
            "f2": np.array([0.0, s.f2]),
            "fCass": np.array([0.0, s.fCass]),
            "f": np.array([0.0, s.f]),
            "Ca_SR": np.array([0.0, s.Ca_SR]),
            "Ca_i": np.array([0.0, s.Ca_i]),
            "Ca_ss": np.array([0.0, s.Ca_ss]),
            "p_iup": np.array([0.0, s.p_iup]),
            "h": np.array([0.0, s.h]),
            "j": np.array([0.0, s.j]),
            "m": np.array([0.0, s.m]),
            "V": np.array([0.0, s.V]),
            "K_i": np.array([0.0, s.K_i]),
            "Xr1": np.array([0.0, s.Xr1]),
            "Xr2": np.array([0.0, s.Xr2]),
            "Xs": np.array([0.0, s.Xs]),
            "Na_i": np.array([0.0, s.Na_i]),
            "r": np.array([0.0, s.r]),
            "s": np.array([0.0, s.s]),
            "v": np.array([0.0, s.v_mech]),
            "w": np.array([0.0, s.w_mech]),
            "N": np.array([0.0, s.N]),
            "A": np.array([0.0, s.A]),
            "l_1": np.array([0.0, s.l_1]),
            "l_2": np.array([0.0, s.l_2]),
            "l_3": np.array([0.0, s.l_3]),
            "R": np.array([0.0, s.R]),
            "O": np.array([0.0, s.O]),
            "I": np.array([0.0, s.I]),
            "RI": np.array([0.0, s.RI]),
            "CaMKt": np.array([0.0, s.CaMKt]),
            "Ca_nSR": np.array([0.0, s.Ca_nSR]),
            "Ca_jSR": np.array([0.0, s.Ca_jSR]),
        }
        s2 = InitialState.from_result(variables)
        assert s2 == s

    def test_v_init_range(self):
        """Потенциал покоя в физиологическом диапазоне."""
        s = InitialState()
        assert -100.0 < s.V < -60.0

    def test_concentrations_positive(self):
        """Все концентрации положительны."""
        s = InitialState()
        assert s.Ca_i > 0
        assert s.Ca_ss > 0
        assert s.Ca_nSR > 0
        assert s.Ca_jSR > 0
        assert s.Na_i > 0
        assert s.K_i > 0

    def test_gate_variables_in_range(self):
        """Все ворота [0, 1]."""
        s = InitialState()
        for name in [
            "d",
            "f2",
            "fCass",
            "f",
            "h",
            "j",
            "m",
            "Xr1",
            "Xr2",
            "Xs",
            "r",
            "s",
        ]:
            val = getattr(s, name)
            assert 0.0 <= val <= 1.0, f"Ворота {name} = {val} вне [0, 1]"


# ===========================================================================
# 3. make_init_state — механическое равновесие
# ===========================================================================


class TestMakeInitState:

    def test_returns_tuple(self, p):
        result = make_init_state(p)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_l0_positive(self, p):
        _, l0 = make_init_state(p)
        assert l0 > 0.0

    def test_mechanical_equilibrium(self, p):
        """В начальном состоянии _fi(l_2) ≈ 0 — механическое равновесие."""
        state, _ = make_init_state(p)
        residual = _fi(state.l_2, p)
        assert abs(residual) < 1e-8

    def test_velocities_zero(self, p):
        """Начальные скорости CE и PE равны нулю."""
        state, _ = make_init_state(p)
        assert state.v_mech == 0.0
        assert state.w_mech == 0.0

    def test_N_in_range(self, p):
        """Доля прикреплённых мостиков в [0, 1]."""
        state, _ = make_init_state(p)
        assert 0.0 <= state.N <= 1.0

    def test_lengths_positive(self, p):
        """Все длины положительны."""
        state, _ = make_init_state(p)
        assert state.l_1 > 0
        assert state.l_2 > 0
        assert state.l_3 > 0

    def test_l0_equals_l2_plus_l3(self, p):
        """l0 = l_2 + l_3 по определению."""
        state, l0 = make_init_state(p)
        assert l0 == pytest.approx(state.l_2 + state.l_3, rel=1e-12)

    def test_base_state_respected(self, p):
        """Электрические переменные из base не перезаписываются."""
        base = InitialState().replace(V=-70.0, Na_i=12.0)
        state, _ = make_init_state(p, base=base)
        assert state.V == -70.0
        assert state.Na_i == 12.0


# ===========================================================================
# 4. rhs() — физические инварианты
# ===========================================================================


class TestRhsPhysics:

    def test_output_shape(self, y0, p, l0):
        dy = rhs(0.0, y0, p, l0=l0)
        assert dy.shape == (33,)

    def test_no_nan_or_inf(self, y0, p, l0):
        dy = rhs(0.0, y0, p, l0=l0)
        assert not np.any(np.isnan(dy)), "rhs() вернул NaN"
        assert not np.any(np.isinf(dy)), "rhs() вернул Inf"

    def test_charge_conservation(self, y0, p, l0):
        """Сохранение заряда: dV + сумма токов / Cm ≈ 0 (без стимула).

        dV = -sum(currents) / 1 (в нормированных единицах).
        Проверяем что dV согласован с током стимула = 0 вне стимула.
        """
        # t = 0 < stim_start (10 мс) — стимул ещё не начался
        dy = rhs(0.0, y0, p, l0=l0)
        elec = _compute_electrical(y0, p, 0.0, _compute_calcium(y0, p))
        dV_direct = elec["dV"]
        assert dy[_I_V] == pytest.approx(dV_direct, rel=1e-10)

    def test_sodium_balance(self, y0, p, l0):
        """dNa_i согласован с токами: i_Na + i_b_Na + 3*i_NaK + 3*i_NaCa."""
        dy = rhs(0.0, y0, p, l0=l0)
        ca = _compute_calcium(y0, p)
        elec = _compute_electrical(y0, p, 0.0, ca)
        expected_dNa = (
            -(
                elec["i_Na"]
                + elec["i_b_Na"]
                + 3.0 * elec["i_NaK"]
                + 3.0 * elec["i_NaCa"]
            )
            * p.Cm
            / (p.V_c * p.F)
        )
        assert dy[_I_NA_I] == pytest.approx(expected_dNa, rel=1e-10)

    def test_ryr_states_sum_conserved(self, y0, p, l0):
        """R + O + I + RI = const. Производная суммы должна быть ≈ 0."""
        dy = rhs(0.0, y0, p, l0=l0)
        from model import _I_R_RYR, _I_O, _I_I_RYR, _I_RI

        d_sum = dy[_I_R_RYR] + dy[_I_O] + dy[_I_I_RYR] + dy[_I_RI]
        assert abs(d_sum) < 1e-12, f"Сумма производных RyR состояний = {d_sum}"

    def test_ca_sr_derivative_zero(self, y0, p, l0):
        """dCa_SR = 0 (переменная заморожена по дизайну, см. комментарий в коде)."""
        dy = rhs(0.0, y0, p, l0=l0)
        from model import _I_CA_SR

        assert dy[_I_CA_SR] == 0.0

    def test_rhs_pure_no_mutation(self, y0, p, l0):
        """rhs() не изменяет входной вектор y."""
        y_copy = y0.copy()
        rhs(0.0, y0, p, l0=l0)
        np.testing.assert_array_equal(y0, y_copy)

    def test_rhs_deterministic(self, y0, p, l0):
        """rhs() детерминирован: два вызова с одним входом дают одинаковый результат."""
        dy1 = rhs(0.0, y0, p, l0=l0)
        dy2 = rhs(0.0, y0, p, l0=l0)
        np.testing.assert_array_equal(dy1, dy2)

    def test_different_params_different_result(self, y0, p, l0):
        """Изменение параметра меняет результат rhs()."""
        p2 = p.replace(g_Na=0.0)
        dy1 = rhs(0.0, y0, p, l0=l0)
        dy2 = rhs(0.0, y0, p2, l0=l0)
        assert not np.allclose(dy1, dy2)

    def test_rhs_parallel_safety(self, y0, p, l0):
        """rhs() можно вызывать параллельно — нет общего состояния."""
        from concurrent.futures import ThreadPoolExecutor

        results = []
        with ThreadPoolExecutor(max_workers=4) as ex:
            futs = [ex.submit(rhs, 0.0, y0.copy(), p, 0.0, l0) for _ in range(8)]
            results = [f.result() for f in futs]
        for r in results[1:]:
            np.testing.assert_array_equal(results[0], r)


# ===========================================================================
# 5. rhs() — численные свойства
# ===========================================================================


class TestRhsNumerics:

    def test_dV_reasonable_magnitude(self, y0, p, l0):
        """dV вне стимула: малое значение (не более 10 мВ/мс)."""
        dy = rhs(0.0, y0, p, l0=l0)
        assert abs(dy[_I_V]) < 10.0, f"|dV| = {abs(dy[_I_V])} слишком велико"

    def test_dCa_i_reasonable(self, y0, p, l0):
        """dCa_i в физиологическом диапазоне."""
        dy = rhs(0.0, y0, p, l0=l0)
        assert abs(dy[_I_CA_I]) < 1.0

    def test_stim_raises_dV(self, y0, p, l0):
        """Во время стимула dV должен быть существенно больше по модулю."""
        t_no_stim = 0.0
        t_stim = p.stim_start + p.stim_duration / 2.0
        dy_no = rhs(t_no_stim, y0, p, l0=l0)
        dy_yes = rhs(t_stim, y0, p, l0=l0)
        assert abs(dy_yes[_I_V]) > abs(dy_no[_I_V])

    def test_no_nan_extreme_ca(self, p, l0):
        """rhs() не падает при крайне малом Ca_i."""
        state, l0 = make_init_state(p)
        y = np.array(state.to_array(), dtype=np.float64)
        y[_I_CA_I] = 1e-9  # почти нулевой Ca_i
        dy = rhs(0.0, y, p, l0=l0)
        assert not np.any(np.isnan(dy))
        assert not np.any(np.isinf(dy))


# ===========================================================================
# 6. Вспомогательные функции механики
# ===========================================================================


class TestMechanicalHelpers:

    def test_p_v_zero_at_minus_vmax(self, p):
        """p(-v_max) = 0."""
        assert _p_v(-p.v_max, p) == pytest.approx(0.0, abs=1e-12)

    def test_p_v_positive_at_zero(self, p):
        """p(0) > 0 (изометрическое сокращение)."""
        assert _p_v(0.0, p) > 0.0

    def test_p_v_zero_at_vmax_lengthening(self, p):
        """p(-v_max) = 0: при максимальной скорости удлинения сила падает до нуля."""
        assert _p_v(-p.v_max, p) == pytest.approx(0.0, abs=1e-12)

    def test_p_v_greater_than_one_during_shortening(self, p):
        """p(v > 0) > 1 в зоне малых скоростей укорочения (нормализованная сила > 1)."""
        assert _p_v(0.001, p) > 1.0

    def test_p_v_less_than_one_during_lengthening(self, p):
        """p(v < 0) < 1 при малых скоростях удлинения."""
        assert _p_v(-0.001, p) < 1.0

    def test_p_prime_v_positive_at_zero(self, p):
        """Производная p'(0) > 0 — монотонность."""
        assert _p_prime_v(0.0, p) > 0.0

    def test_p_prime_v_finite(self, p):
        """p'(v) конечна в нескольких точках."""
        for v in [-p.v_max * 2, -p.v_max, -0.001, 0.0, 0.001, p.v_max]:
            val = _p_prime_v(v, p)
            assert np.isfinite(val), f"p'({v}) = {val}"

    def test_pi_N_A_limits(self, p):
        """pi_N_A: при N→0 возвращает 1, при N→∞ возвращает pi_min."""
        A = 1e-3
        assert _pi_N_A(0.0, A, p) == pytest.approx(1.0)
        assert _pi_N_A(1e6, A, p) == pytest.approx(p.pi_min)

    def test_N0_in_range(self, p):
        """N0(l_2) в [0, 1]."""
        state, _ = make_init_state(p)
        n = _N0(state.l_2, p)
        assert 0.0 <= n <= 1.0

    def test_mechanical_forces_non_negative_at_equilibrium(self, p):
        """F_SE, F_PE, F_XSE >= 0 в равновесном состоянии."""
        state, l0 = make_init_state(p)
        y = np.array(state.to_array(), dtype=np.float64)
        mech = _compute_mechanical(y, p, F_afterload=0.0, l0=l0)
        assert mech["F_SE"] >= 0.0
        assert mech["F_PE"] >= 0.0
        assert mech["F_XSE"] >= 0.0


# ===========================================================================
# 7. _gate_CaL — зависимость fCass от Ca_ss, а не от V
# ===========================================================================


class TestGateCaL:

    def test_fcass_depends_on_ca_ss_not_V(self):
        """dfCass должен меняться при изменении Ca_ss, но не при изменении V."""
        V = -85.0
        Ca_ss_low = 0.0001
        Ca_ss_high = 0.001
        fCass = 0.9

        _, _, dfCass_low, _ = _gate_CaL(V, Ca_ss_low, 0.01, 0.99, fCass, 0.98)
        _, _, dfCass_high, _ = _gate_CaL(V, Ca_ss_high, 0.01, 0.99, fCass, 0.98)
        assert dfCass_low != dfCass_high, "dfCass не зависит от Ca_ss — ошибка"

    def test_fcass_independent_of_V(self):
        """При фиксированном Ca_ss dfCass не зависит от V."""
        Ca_ss = 0.0002
        fCass = 0.9
        _, _, dfCass_v1, _ = _gate_CaL(-85.0, Ca_ss, 0.01, 0.99, fCass, 0.98)
        _, _, dfCass_v2, _ = _gate_CaL(-40.0, Ca_ss, 0.01, 0.99, fCass, 0.98)
        assert dfCass_v1 == pytest.approx(dfCass_v2, rel=1e-12)

    def test_gate_CaL_returns_four_values(self):
        dd, df2, dfCass, df = _gate_CaL(-85.0, 0.0002, 0.01, 0.99, 0.9, 0.98)
        assert all(np.isfinite(v) for v in [dd, df2, dfCass, df])

    def test_gate_d_inf_at_high_V(self):
        """При высоком V d должен стремиться к 1."""
        dd_high, *_ = _gate_CaL(60.0, 0.0002, 0.0, 0.99, 0.9, 0.98)
        dd_low, *_ = _gate_CaL(-85.0, 0.0002, 0.0, 0.99, 0.9, 0.98)
        # d_inf больше при высоком V → dd > 0 при d=0 и высоком V
        assert dd_high > dd_low


# ===========================================================================
# 8. calculate_outputs — соответствие rhs()
# ===========================================================================


class TestCalculateOutputs:

    def test_output_keys(self, y0, p, l0):
        """calculate_outputs возвращает все ожидаемые ключи."""
        t_arr = np.array([0.0])
        y_arr = y0[np.newaxis, :]
        out = calculate_outputs(t_arr, y_arr, p, l0=l0)

        expected_currents = [
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
        expected_forces = ["F_CE", "F_SE", "F_PE", "F_VS1", "F_VS2", "F_XSE"]
        for k in expected_currents + expected_forces:
            assert k in out, f"Ключ '{k}' отсутствует в calculate_outputs"

    def test_output_length(self, y0, p, l0):
        """Длина каждого массива соответствует числу точек."""
        N = 5
        t_arr = np.linspace(0.0, 4.0, N)
        y_arr = np.tile(y0, (N, 1))
        out = calculate_outputs(t_arr, y_arr, p, l0=l0)
        for k, v in out.items():
            assert len(v) == N, f"Длина '{k}' = {len(v)}, ожидалось {N}"

    def test_currents_consistent_with_compute(self, y0, p, l0):
        """Токи из calculate_outputs совпадают с _compute_electrical."""
        t_arr = np.array([0.0])
        y_arr = y0[np.newaxis, :]
        out = calculate_outputs(t_arr, y_arr, p, l0=l0)

        ca = _compute_calcium(y0, p)
        elec = _compute_electrical(y0, p, 0.0, ca)

        assert out["i_Na"][0] == pytest.approx(elec["i_Na"], rel=1e-12)
        assert out["i_CaL"][0] == pytest.approx(elec["i_CaL"], rel=1e-12)
        assert out["i_NaK"][0] == pytest.approx(elec["i_NaK"], rel=1e-12)

    def test_forces_consistent_with_compute(self, y0, p, l0):
        """Силы из calculate_outputs совпадают с _compute_mechanical."""
        t_arr = np.array([0.0])
        y_arr = y0[np.newaxis, :]
        out = calculate_outputs(t_arr, y_arr, p, l0=l0)

        mech = _compute_mechanical(y0, p, 0.0, l0)
        assert out["F_CE"][0] == pytest.approx(mech["F_CE"], rel=1e-12)
        assert out["F_SE"][0] == pytest.approx(mech["F_SE"], rel=1e-12)

    def test_no_nan_in_outputs(self, y0, p, l0):
        t_arr = np.array([0.0])
        y_arr = y0[np.newaxis, :]
        out = calculate_outputs(t_arr, y_arr, p, l0=l0)
        for k, v in out.items():
            assert not np.any(np.isnan(v)), f"NaN в '{k}'"


# ===========================================================================
# 9. Стимул — периодичность и амплитуда
# ===========================================================================


class TestStimCurrent:

    def test_no_stim_before_start(self, p):
        assert _stim_current(0.0, p) == 0.0

    def test_stim_during_pulse(self, p):
        t_mid = p.stim_start + p.stim_duration / 2.0
        assert _stim_current(t_mid, p) == -p.stim_amplitude

    def test_no_stim_after_pulse(self, p):
        t_after = p.stim_start + p.stim_duration + 1.0
        assert _stim_current(t_after, p) == 0.0

    def test_stim_periodicity(self, p):
        """Стимул повторяется с периодом stim_period."""
        t_mid = p.stim_start + p.stim_duration / 2.0
        for k in range(1, 4):
            t = t_mid + k * p.stim_period
            assert _stim_current(t, p) == pytest.approx(-p.stim_amplitude)

    def test_stim_amplitude_in_rhs(self, y0, p, l0):
        """dV во время стимула отличается от dV без стимула ровно на amplitude."""
        t_no = 0.0
        t_stim = p.stim_start + p.stim_duration / 2.0
        dy_no = rhs(t_no, y0, p, l0=l0)
        dy_yes = rhs(t_stim, y0, p, l0=l0)
        # dV_stim = dV_no - (-amplitude) = dV_no + amplitude
        assert dy_yes[_I_V] == pytest.approx(dy_no[_I_V] + p.stim_amplitude, rel=1e-10)


# ===========================================================================
# 10. Параметрический sweep — make_init_state устойчив к вариации параметров
# ===========================================================================


class TestSweepReadiness:

    @pytest.mark.parametrize("g_CaL_scale", [0.5, 1.0, 1.5, 2.0])
    def test_rhs_valid_for_scaled_g_CaL(self, g_CaL_scale):
        """rhs() не падает при масштабировании g_CaL."""
        p = TNNPMParams().replace(g_CaL=TNNPMParams().g_CaL * g_CaL_scale)
        state, l0 = make_init_state(p)
        y = np.array(state.to_array(), dtype=np.float64)
        dy = rhs(0.0, y, p, l0=l0)
        assert not np.any(np.isnan(dy))
        assert not np.any(np.isinf(dy))

    @pytest.mark.parametrize("stim_hz", [0.5, 1.0, 2.0])
    def test_rhs_valid_for_different_frequencies(self, stim_hz):
        """rhs() корректен при разных частотах стимуляции."""
        p = TNNPMParams().replace(stim_period=1000.0 / stim_hz)
        state, l0 = make_init_state(p)
        y = np.array(state.to_array(), dtype=np.float64)
        dy = rhs(0.0, y, p, l0=l0)
        assert not np.any(np.isnan(dy))
