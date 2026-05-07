"""
model.py
========
Математическое ядро модели TNNPM (Ten Tusscher – Niederer – Noble – Panfilov +
механика ЭКБ + RyR Shanon + CaMK).

Публичный API модуля:
    rhs(t, y, p, F_afterload, l0) -> np.ndarray
        Правая часть системы ОДУ. Передаётся в солвер.

    calculate_outputs(t_arr, y_arr, p, F_afterload, l0) -> dict
        Вычисляет токи и силы по уже решённой траектории.
        Использует те же приватные функции что и rhs(), без дублирования.

    make_init_state(p) -> InitialState
        Вычисляет самосогласованные начальные условия для механики
        методом бисекции (заменяет delenie() + calculate_init_conditions()).

Принципы:
    - rhs() — чистая функция: нет изменяемого состояния, нет self,
      нет записи в атрибуты. Безопасна при параллельных прогонах.
    - Вся общая математика вынесена в _compute_calcium(), _compute_electrical(),
      _compute_mechanical(). rhs() и calculate_outputs() вызывают их совместно,
      исключая дублирование формул.
    - np.float128 убран: солвер работает в float64, конвертация не нужна.
    - F_afterload и l0 — параметры режима нагрузки, передаются явно,
      не хранятся в параметрах модели.
"""

from __future__ import annotations

import numpy as np
from numpy import exp, log, sqrt, floor
from scipy.optimize import brentq

from parameters import InitialState, TNNPMParams

# ---------------------------------------------------------------------------
# Индексы вектора состояния y (совпадают с порядком в InitialState.to_array)
# ---------------------------------------------------------------------------
_I_D = 0
_I_F2 = 1
_I_FCASS = 2
_I_F = 3
_I_CA_SR = 4  # dCa_SR = 0, переменная сохранена для совместимости
_I_CA_I = 5
_I_CA_SS = 6
_I_P_IUP = 7
_I_H = 8
_I_J = 9
_I_M = 10
_I_V = 11
_I_K_I = 12
_I_XR1 = 13
_I_XR2 = 14
_I_XS = 15
_I_NA_I = 16
_I_R_gate = 17  # ворота r (i_to), не путать с R (RyR)
_I_S = 18
_I_V_MECH = 19  # скорость CE
_I_W_MECH = 20  # скорость PE
_I_N = 21
_I_A = 22
_I_L1 = 23
_I_L2 = 24
_I_L3 = 25
_I_R_RYR = 26
_I_O = 27
_I_I_RYR = 28
_I_RI = 29
_I_CAMKT = 30
_I_CA_NSR = 31
_I_CA_JSR = 32

_N_STATES = 33


# ---------------------------------------------------------------------------
# Вспомогательные скалярные функции механической части
# (чистые функции от v, l_1 и параметров p)
# ---------------------------------------------------------------------------


def _chi(v: float, p: TNNPMParams) -> float:
    """Кинетический коэффициент chi(v)."""
    if v <= 0.0:
        return p.chi_1 + p.chi_2 * v / p.v_max
    return p.chi_1


def _q_v(v: float, p: TNNPMParams) -> float:
    """Скоростная зависимость q(v)."""
    if v <= 0.0:
        return p.q_1 - p.q_2 * v / p.v_max
    elif v <= p.x_st * p.v_max:
        return (p.q_4 - p.q_3) * v / (p.x_st * p.v_max) + p.q_3
    else:
        return p.q_4 / (1.0 + p.beta_Q * (v / p.v_max - p.x_st)) ** p.alpha_Q


def _P_star(v: float, p: TNNPMParams) -> float:
    gamma = p.a * p.d_h * p.v_1**2.0 / (3.0 * p.a * p.d_h - (p.a + 1.0) * p.v_1)
    return (
        1.0
        + p.d_h
        - p.d_h**2.0
        * p.a
        / ((p.a + 1.0) * v + p.d_h * p.a + p.a * p.d_h * v**2.0 / gamma)
    )


def _G_star(v: float, p: TNNPMParams) -> float:
    """Нормированная сила мостиков G*(v/v_max)."""
    den = (0.4 * p.a + 1.0) * v / p.a + 1.0
    if v <= 0.0:
        return 1.0 + 0.6 * v
    elif v <= p.v_1:
        return _P_star(v, p) / den
    else:
        return _P_star(v, p) * exp(-p.alpha_G * (v - p.v_1) ** p.alpha_P) / den


def _p_v(v: float, p: TNNPMParams) -> float:
    """Нормированная сила CE как функция скорости."""
    if v <= -p.v_max:
        return 0.0
    elif v <= 0.0:
        return (
            p.a
            * (1.0 + v / p.v_max)
            / ((p.a - v / p.v_max) * (1.0 + 0.6 * v / p.v_max))
        )
    elif v <= p.v_1 * p.v_max:
        return (0.4 * p.a + 1.0) * v / (p.a * p.v_max) + 1.0
    else:
        return ((0.4 * p.a + 1.0) * v / (p.a * p.v_max) + 1.0) * exp(
            p.alpha_G * (v / p.v_max - p.v_1) ** p.alpha_P
        )


def _p_prime_v(v: float, p: TNNPMParams) -> float:
    """Производная p(v) по v."""
    if v <= -p.v_max:
        return p.a * (0.4 + 0.4 * p.a) / (p.v_max * ((p.a + 1.0) * 0.4) ** 2.0)
    elif v <= 0.0:
        return (
            p.a
            * (1.0 + 0.4 * p.a + 1.2 * v / p.v_max + 0.6 * (v / p.v_max) ** 2.0)
            / (p.v_max * ((p.a - v / p.v_max) * (1.0 + 0.6 * v / p.v_max)) ** 2.0)
        )
    elif v <= p.v_1 * p.v_max:
        return (0.4 * p.a + 1.0) / (p.a * p.v_max)
    else:
        return (
            exp(p.alpha_G * (v / p.v_max - p.v_1) ** p.alpha_P)
            * (
                (0.4 * p.a + 1.0) / p.a
                + p.alpha_G
                * p.alpha_P
                * (1.0 + (0.4 * p.a + 1.0) * v / (p.a * p.v_max))
                * (v / p.v_max - p.v_1) ** (p.alpha_P - 1.0)
            )
            / p.v_max
        )


def _k_p_v(v: float, p: TNNPMParams) -> float:
    return _chi(v, p) * p.chi_0 * _q_v(v, p) * p.m_0 * _G_star(v / p.v_max, p)


def _k_m_v(v: float, p: TNNPMParams) -> float:
    return p.chi_0 * _q_v(v, p) * (1.0 - _chi(v, p) * p.m_0 * _G_star(v / p.v_max, p))


def _M(A: float, p: TNNPMParams) -> float:
    ratio = A / p.A_tot
    return ratio**p.mu * (1.0 + p.k_mu**p.mu) / (ratio**p.mu + p.k_mu**p.mu)


def _n_1(l_1: float, p: TNNPMParams) -> float:
    w1 = (p.g_1 * l_1 + p.g_2) * (
        p.n1_A
        + (p.n1_K - p.n1_A) / (p.n1_C + p.n1_Q * exp(-p.n1_B * l_1)) ** (1.0 / p.n1_nu)
    )
    return max(0.0, min(1.0, w1))


def _L_oz(l_1: float, p: TNNPMParams) -> float:
    if l_1 <= p.s055:
        return (l_1 + p.s_0) / (p.s046 + p.s_0)
    return (p.s_0 + p.s055) / (p.s046 + p.s_0)


def _pi_N_A(N: float, A: float, p: TNNPMParams) -> float:
    N_A = p.A_tot * p.s_c * N / A
    if N_A <= 0.0:
        return 1.0
    elif N_A <= 1.0:
        return p.pi_min**N_A
    return p.pi_min


# ---------------------------------------------------------------------------
# Приватные вычислительные блоки
# ---------------------------------------------------------------------------


def _compute_calcium(y: np.ndarray, p: TNNPMParams) -> dict:
    """Кальциевая динамика: CaMK, SERCA, RyR, буферные факторы, токи Ca.

    Возвращает dict со всеми промежуточными величинами нужными как для
    производных, так и для calculate_outputs().
    """
    Ca_i = y[_I_CA_I]
    Ca_ss = y[_I_CA_SS]
    Ca_nSR = y[_I_CA_NSR]
    Ca_jSR = y[_I_CA_JSR]
    p_iup = y[_I_P_IUP]
    O = y[_I_O]

    # CaMK
    CaMKb = p.CaMKo * (1.0 - y[_I_CAMKT]) / (1.0 + p.KmCaM / Ca_ss)
    CaMKa = CaMKb + y[_I_CAMKT]
    fJupp = 1.0 / (1.0 + p.KmCaMK / CaMKa)
    dCaMKt = p.aCaMK * CaMKb * (CaMKb + y[_I_CAMKT]) - p.bCaMK * y[_I_CAMKT]

    # SERCA (i_up)
    Jupnp = p.Vmax_up / (1.0 + p.K_up**2.0 / Ca_i**2.0)
    Jupp = 2.75 * p.Vmax_up / (1.0 + (p.K_up - 0.00017) ** 2.0 / Ca_i**2.0)
    i_up = (1.0 - fJupp) * Jupnp + fJupp * Jupp

    # i_leak
    i_leak = p.V_leak * (Ca_nSR - Ca_i)

    # i_tr (nSR → jSR)
    i_tr = (Ca_nSR - Ca_jSR) / 60.0

    # dCa_nSR
    dCa_nSR = i_up - i_leak - i_tr * p.V_jSR / p.V_nSR

    # Semin: p_iup
    dp_iup = p.k_sm_p * (Ca_i**2.0 * (1.0 - p_iup) - p.K_lrg_p**2.0 * p_iup)

    # RyR (Shanon)
    k_CaSR = p.max_sr - (p.max_sr - p.min_sr) / (1.0 + (p.EC / Ca_jSR) ** 2.0)
    k_o_SR_Ca = p.k_o_Ca / k_CaSR
    k_i_SR_Ca = p.k_i_Ca * k_CaSR

    R_ryr = y[_I_R_RYR]
    I_ryr = y[_I_I_RYR]
    RI = y[_I_RI]

    dR = (
        p.k_im * RI
        - k_i_SR_Ca * R_ryr * Ca_ss
        - k_o_SR_Ca * R_ryr * Ca_ss**2.0
        + p.k_om * O
    )
    dO = (
        k_o_SR_Ca * R_ryr * Ca_ss**2.0
        - p.k_om * O
        - k_i_SR_Ca * O * Ca_ss
        + p.k_im * I_ryr
    )
    dI = (
        k_i_SR_Ca * O * Ca_ss
        - p.k_im * I_ryr
        - p.k_om * I_ryr
        + k_o_SR_Ca * RI * Ca_ss**2.0
    )
    dRI = (
        p.k_om * I_ryr
        - k_o_SR_Ca * RI * Ca_ss**2.0
        - p.k_im * RI
        + k_i_SR_Ca * R_ryr * Ca_ss
    )

    # i_rel, i_xfer
    i_rel = p.V_rel * O * (Ca_jSR - Ca_ss)
    i_xfer = p.V_xfer * (Ca_ss - Ca_i)

    # dCa_jSR (с буфером)
    Ca_jsr_buf = 1.0 / (1.0 + p.Buf_jsr * p.K_buf_jsr / (Ca_jSR + p.K_buf_jsr) ** 2.0)
    dCa_jSR = Ca_jsr_buf * (i_tr - i_rel)

    return dict(
        CaMKb=CaMKb,
        CaMKa=CaMKa,
        fJupp=fJupp,
        dCaMKt=dCaMKt,
        i_up=i_up,
        i_leak=i_leak,
        i_tr=i_tr,
        i_rel=i_rel,
        i_xfer=i_xfer,
        dp_iup=dp_iup,
        dR=dR,
        dO=dO,
        dI=dI,
        dRI=dRI,
        dCa_nSR=dCa_nSR,
        dCa_jSR=dCa_jSR,
    )


def _compute_electrical(y: np.ndarray, p: TNNPMParams, t: float, ca: dict) -> dict:
    """Электрическая часть: все токи, ворота, dV, dNa_i, dK_i.

    Принимает результат _compute_calcium() через аргумент ca, чтобы
    не пересчитывать i_up, i_leak, i_rel и т.д.
    """
    V = y[_I_V]
    Na_i = y[_I_NA_I]
    K_i = y[_I_K_I]
    Ca_i = y[_I_CA_I]
    Ca_ss = y[_I_CA_SS]
    d = y[_I_D]
    f2 = y[_I_F2]
    fCass = y[_I_FCASS]
    f = y[_I_F]
    h = y[_I_H]
    j = y[_I_J]
    m = y[_I_M]
    Xr1 = y[_I_XR1]
    Xr2 = y[_I_XR2]
    Xs = y[_I_XS]
    r_gate = y[_I_R_gate]
    s = y[_I_S]
    A = y[_I_A]

    RT_F = p.R * p.T / p.F

    # Равновесные потенциалы
    E_Na = RT_F * log(p.Na_o / Na_i)
    E_K = RT_F * log(p.K_o / K_i)
    E_Ca = 0.5 * RT_F * log(p.Ca_o / Ca_i)
    E_Ks = RT_F * log((p.K_o + p.P_kna * p.Na_o) / (K_i + p.P_kna * Na_i))

    # i_Na
    i_Na = p.g_Na * m**3.0 * h * j * (V - E_Na)

    # i_CaL
    exp2 = exp(2.0 * (V - 15.0) * p.F / (p.R * p.T))
    i_CaL = (
        p.g_CaL
        * d
        * f
        * f2
        * fCass
        * 4.0
        * (V - 15.0)
        * p.F**2.0
        / (p.R * p.T)
        * (0.25 * Ca_ss * exp2 - p.Ca_o)
        / (exp2 - 1.0)
    )

    # i_to, i_Kr, i_Ks, i_K1
    i_to = p.g_to * r_gate * s * (V - E_K)
    i_Kr = p.g_Kr * sqrt(p.K_o / 5.4) * Xr1 * Xr2 * (V - E_K)
    i_Ks = p.g_Ks * Xs**2.0 * (V - E_Ks)

    alpha_K1 = 0.1 / (1.0 + exp(0.06 * (V - E_K - 200.0)))
    beta_K1 = (3.0 * exp(0.0002 * (V - E_K + 100.0)) + exp(0.1 * (V - E_K - 10.0))) / (
        1.0 + exp(-0.5 * (V - E_K))
    )
    xK1_inf = alpha_K1 / (alpha_K1 + beta_K1)
    i_K1 = p.g_K1 * xK1_inf * sqrt(p.K_o / 5.4) * (V - E_K)

    # i_K_ATP
    PATP = 1.0 / (1.0 + (p.ATPi / p.KmATP) ** 2.2)
    i_K_ATP = p.g_K_ATP * PATP * (p.K_o / 5.4) ** 0.24 * (V - E_K)

    # i_NaK
    i_NaK = (
        p.P_NaK
        * p.K_o
        * Na_i
        / (
            (p.K_o + p.K_mk)
            * (Na_i + p.K_mNa)
            * (
                1.0
                + 0.1245 * exp(-0.1 * V * p.F / (p.R * p.T))
                + 0.0353 * exp(-V * p.F / (p.R * p.T))
            )
        )
    )

    # i_NaCa
    VF_RT = V * p.F / (p.R * p.T)
    i_NaCa = (
        p.K_NaCa
        * (
            exp(p.gamma * VF_RT) * Na_i**3.0 * p.Ca_o
            - exp((p.gamma - 1.0) * VF_RT) * p.Na_o**3.0 * Ca_i * p.alpha_NaCa
        )
        / (
            (p.Km_Nai**3.0 + p.Na_o**3.0)
            * (p.Km_Ca + p.Ca_o)
            * (1.0 + p.K_sat * exp((p.gamma - 1.0) * VF_RT))
        )
    )

    # i_b_Ca, i_p_Ca, i_b_Na, i_p_K
    i_b_Ca = p.g_bCa * (V - E_Ca)
    i_p_Ca = p.g_pCa * Ca_i / (Ca_i + p.K_pCa)
    i_b_Na = p.g_bna * (V - E_Na)
    i_p_K = p.g_pK * (V - E_K) / (1.0 + exp((25.0 - V) / 5.98))

    # Стимул
    i_Stim = _stim_current(t, p)

    # dV
    dV = -(
        i_K1
        + i_to
        + i_Kr
        + i_Ks
        + i_CaL
        + i_NaK
        + i_Na
        + i_b_Na
        + i_NaCa
        + i_b_Ca
        + i_p_K
        + i_p_Ca
        + i_K_ATP
        + i_Stim
    )

    # dK_i
    dK_i = (
        -(i_K1 + i_to + i_Kr + i_Ks + i_p_K + i_Stim - 2.0 * i_NaK + i_K_ATP)
        * p.Cm
        / (p.V_c * p.F)
    )

    # dNa_i
    dNa_i = -(i_Na + i_b_Na + 3.0 * i_NaK + 3.0 * i_NaCa) * p.Cm / (p.V_c * p.F)

    # dCa_i (включает dA из механики — передаётся ниже через rhs)
    i_up = ca["i_up"]
    i_leak = ca["i_leak"]
    i_xfer = ca["i_xfer"]

    Ca_i_bufc = 1.0 / (1.0 + p.Buf_c * p.K_buf_c / (Ca_i + p.K_buf_c) ** 2.0)

    # dCa_ss (без dA — dA влияет только на dCa_i)
    Ca_ss_bufss = 1.0 / (1.0 + p.Buf_ss * p.K_buf_ss / (Ca_ss + p.K_buf_ss) ** 2.0)
    dCa_ss = Ca_ss_bufss * (
        -i_CaL * p.Cm / (2.0 * p.V_ss * p.F)
        + ca["i_rel"] * p.V_jSR / p.V_ss
        - i_xfer * p.V_c / p.V_ss
    )

    # Ворота
    dd, df2, dfCass, df = _gate_CaL(V, Ca_ss, d, f2, fCass, f)
    dh, dj, dm = _gate_Na(V, h, j, m)
    dr, ds = _gate_to(V, r_gate, s)
    dXr1, dXr2 = _gate_Kr(V, Xr1, Xr2)
    dXs = _gate_Ks(V, Xs)

    # dA (связывание Ca с TnC) — нужен для dCa_i
    A_off = p.a_off * _pi_N_A(y[_I_N], A, p) * exp(-p.k_A * A)
    dA = p.a_on * (p.A_tot - A) * Ca_i - A_off * A

    # dCa_i (с dA)
    dCa_i = Ca_i_bufc * (
        (i_leak - i_up) * p.V_nSR / p.V_c
        + i_xfer
        - (i_b_Ca + i_p_Ca - 2.0 * i_NaCa) * p.Cm / (2.0 * p.V_c * p.F)
        - dA
    )

    return dict(
        # токи (для calculate_outputs)
        i_Na=i_Na,
        i_CaL=i_CaL,
        i_NaCa=i_NaCa,
        i_NaK=i_NaK,
        i_K1=i_K1,
        i_Kr=i_Kr,
        i_Ks=i_Ks,
        i_K_ATP=i_K_ATP,
        i_to=i_to,
        i_b_Ca=i_b_Ca,
        i_p_Ca=i_p_Ca,
        i_b_Na=i_b_Na,
        # производные
        dV=dV,
        dK_i=dK_i,
        dNa_i=dNa_i,
        dCa_i=dCa_i,
        dCa_ss=dCa_ss,
        dA=dA,
        dd=dd,
        df2=df2,
        dfCass=dfCass,
        df=df,
        dh=dh,
        dj=dj,
        dm=dm,
        dr=dr,
        ds=ds,
        dXr1=dXr1,
        dXr2=dXr2,
        dXs=dXs,
    )


def _compute_mechanical(
    y: np.ndarray, p: TNNPMParams, F_afterload: float, l0: float
) -> dict:
    """Механическая часть (ЭКБ-модель): производные длин, скоростей, N; силы."""
    v_m = y[_I_V_MECH]
    w_m = y[_I_W_MECH]
    N = y[_I_N]
    A = y[_I_A]
    l_1 = y[_I_L1]
    l_2 = y[_I_L2]
    l_3 = y[_I_L3]

    K_chi = (
        _k_p_v(v_m, p) * _M(A, p) * _n_1(l_1, p) * _L_oz(l_1, p) * (1.0 - N)
        - _k_m_v(v_m, p) * N
    )
    dN = K_chi

    F_muscle = p.beta_3 * (exp(p.alpha_3 * l_3) - 1.0)
    l_total = l_2 + l_3

    isotonic_mode = (
        F_afterload > 1e-15
        and F_muscle >= F_afterload
        and l_total <= l0 * (1.0 + 1.0e-4)
        and F_muscle > p.r0
    )

    # Вязкие коэффициенты
    if v_m <= 0.0:
        alpha_p = p.alpha_vp_l
        k_P_vis = p.beta_vp_l * exp(p.alpha_vp_l * l_1)
    else:
        alpha_p = p.alpha_vp_s
        k_P_vis = p.beta_vp_s * exp(p.alpha_vp_s * l_1)

    if w_m <= v_m:
        alpha_s = p.alpha_vs_l
        k_S_vis = p.beta_vs_l * exp(p.alpha_vs_l * (l_2 - l_1))
    else:
        alpha_s = p.alpha_vs_s
        k_S_vis = p.beta_vs_s * exp(p.alpha_vs_s * (l_2 - l_1))

    # phi_chi = dv
    PE_term = p.alpha_2 * p.beta_2 * exp(p.alpha_2 * l_2)
    XSE_term = p.alpha_3 * p.beta_3 * exp(p.alpha_3 * l_3)

    if isotonic_mode:
        phi_chi = -(
            p.llambda * K_chi * _p_v(v_m, p)
            + alpha_p * k_P_vis * v_m**2.0
            + PE_term * w_m
        ) / (p.llambda * N * _p_prime_v(v_m, p) + k_P_vis)
    else:
        phi_chi = -(
            p.llambda * K_chi * _p_v(v_m, p)
            + alpha_p * k_P_vis * v_m**2.0
            + (PE_term + XSE_term) * w_m
        ) / (p.llambda * N * _p_prime_v(v_m, p) + k_P_vis)

    SE_term = p.alpha_1 * p.beta_1 * exp(p.alpha_1 * (l_2 - l_1))

    if isotonic_mode:
        dw = (
            phi_chi
            - alpha_s * (w_m - v_m) ** 2.0
            - (SE_term * (w_m - v_m) + PE_term * w_m) / k_S_vis
        )
        dl_3 = 0.0
    else:
        dw = (
            phi_chi
            - alpha_s * (w_m - v_m) ** 2.0
            - (SE_term * (w_m - v_m) + (PE_term + XSE_term) * w_m) / k_S_vis
        )
        dl_3 = -w_m

    # Силы
    F_CE = p.llambda * _p_v(v_m, p) * N
    F_SE = p.beta_1 * (exp(p.alpha_1 * (l_2 - l_1)) - 1.0)
    F_PE = p.beta_2 * (exp(p.alpha_2 * l_2) - 1.0)
    F_VS1 = k_P_vis * v_m
    F_VS2 = k_S_vis * (w_m - v_m)
    F_XSE = p.beta_3 * (exp(p.alpha_3 * l_3) - 1.0)

    return dict(
        dN=dN,
        dv=phi_chi,
        dw=dw,
        dl_1=v_m,
        dl_2=w_m,
        dl_3=dl_3,
        F_CE=F_CE,
        F_SE=F_SE,
        F_PE=F_PE,
        F_VS1=F_VS1,
        F_VS2=F_VS2,
        F_XSE=F_XSE,
    )


# ---------------------------------------------------------------------------
# Ворота — отдельные функции для читаемости
# ---------------------------------------------------------------------------


def _gate_CaL(V, Ca_ss, d, f2, fCass, f):
    d_inf = 1.0 / (1.0 + exp((-8.0 - V) / 7.5))
    tau_d = (1.4 / (1.0 + exp((-35.0 - V) / 13.0)) + 0.25) * (
        1.4 / (1.0 + exp((V + 5.0) / 5.0))
    ) + 1.0 / (1.0 + exp((50.0 - V) / 20.0))
    f2_inf = 0.67 / (1.0 + exp((V + 35.0) / 7.0)) + 0.33
    tau_f2 = (
        562.0 * exp(-((V + 27.0) ** 2.0) / 240.0)
        + 31.0 / (1.0 + exp((25.0 - V) / 10.0))
        + 80.0 / (1.0 + exp((V + 30.0) / 10.0))
    )
    fCass_inf = 0.6 / (1.0 + (Ca_ss / 0.05) ** 2.0) + 0.4
    tau_fCass = 80.0 / (1.0 + (Ca_ss / 0.05) ** 2.0) + 2.0
    f_inf = 1.0 / (1.0 + exp((V + 20.0) / 7.0))
    tau_f = (
        1102.5 * exp(-((V + 27.0) ** 2.0) / 225.0)
        + 200.0 / (1.0 + exp((13.0 - V) / 10.0))
        + 180.0 / (1.0 + exp((V + 30.0) / 10.0))
        + 20.0
    )
    return (d_inf - d) / tau_d, (f2_inf - f2) / tau_f2, (fCass_inf - fCass) / tau_fCass, (f_inf - f) / tau_f


def _gate_Na(V, h, j, m):
    h_inf = 1.0 / (1.0 + exp((V + 71.55) / 7.43)) ** 2.0
    alpha_h = 0.057 * exp(-(V + 80.0) / 6.8) if V < -40.0 else 0.0
    beta_h = (
        (2.7 * exp(0.079 * V) + 310000.0 * exp(0.3485 * V))
        if V < -40.0
        else 0.77 / (0.13 * (1.0 + exp((V + 10.66) / -11.1)))
    )
    tau_h = 1.0 / (alpha_h + beta_h)

    j_inf = h_inf
    alpha_j = (
        (
            (-25428.0 * exp(0.2444 * V) - 6.948e-6 * exp(-0.04391 * V))
            * (V + 37.78)
            / (1.0 + exp(0.311 * (V + 79.23)))
        )
        if V < -40.0
        else 0.0
    )
    beta_j = (
        (0.02424 * exp(-0.01052 * V) / (1.0 + exp(-0.1378 * (V + 40.14))))
        if V < -40.0
        else 0.6 * exp(0.057 * V) / (1.0 + exp(-0.1 * (V + 32.0)))
    )
    tau_j = 1.0 / (alpha_j + beta_j)

    m_inf = 1.0 / (1.0 + exp((-56.86 - V) / 9.03)) ** 2.0
    tau_m = (1.0 / (1.0 + exp((-60.0 - V) / 5.0))) * (
        0.1 / (1.0 + exp((V + 35.0) / 5.0)) + 0.1 / (1.0 + exp((V - 50.0) / 200.0))
    )

    return (h_inf - h) / tau_h, (j_inf - j) / tau_j, (m_inf - m) / tau_m


def _gate_to(V, r_gate, s):
    r_inf = 1.0 / (1.0 + exp((20.0 - V) / 6.0))
    tau_r = 9.5 * exp(-((V + 40.0) ** 2.0) / 1800.0) + 0.8
    s_inf = 1.0 / (1.0 + exp((V + 20.0) / 5.0))
    tau_s = (
        85.0 * exp(-((V + 45.0) ** 2.0) / 320.0)
        + 5.0 / (1.0 + exp((V - 20.0) / 5.0))
        + 3.0
    )
    return (r_inf - r_gate) / tau_r, (s_inf - s) / tau_s


def _gate_Kr(V, Xr1, Xr2):
    xr1_inf = 1.0 / (1.0 + exp((-26.0 - V) / 7.0))
    tau_xr1 = (450.0 / (1.0 + exp((-45.0 - V) / 10.0))) * (
        6.0 / (1.0 + exp((V + 30.0) / 11.5))
    )
    xr2_inf = 1.0 / (1.0 + exp((V + 88.0) / 24.0))
    tau_xr2 = (3.0 / (1.0 + exp((-60.0 - V) / 20.0))) * (
        1.12 / (1.0 + exp((V - 60.0) / 20.0))
    )
    return (xr1_inf - Xr1) / tau_xr1, (xr2_inf - Xr2) / tau_xr2


def _gate_Ks(V, Xs):
    xs_inf = 1.0 / (1.0 + exp((-5.0 - V) / 14.0))
    tau_xs = (1400.0 / sqrt(1.0 + exp((5.0 - V) / 6.0))) * (
        1.0 / (1.0 + exp((V - 35.0) / 15.0))
    ) + 80.0
    return (xs_inf - Xs) / tau_xs


def _stim_current(t: float, p: TNNPMParams) -> float:
    """Стимул: прямоугольный импульс с периодом stim_period."""
    t_in_cycle = t - floor(t / p.stim_period) * p.stim_period
    if p.stim_start <= t_in_cycle <= p.stim_start + p.stim_duration:
        return -p.stim_amplitude
    return 0.0


# ---------------------------------------------------------------------------
# Публичный API
# ---------------------------------------------------------------------------


def rhs(
    t: float,
    y: np.ndarray,
    p: TNNPMParams,
    F_afterload: float = 0.0,
    l0: float | None = None,
) -> np.ndarray:
    """Правая часть системы ОДУ модели TNNPM.

    Аргументы:
        t           — текущее время (мс)
        y           — вектор состояния (33 элемента)
        p           — параметры модели (TNNPMParams)
        F_afterload — постнагрузка (мН), 0 = изометрический режим
        l0          — начальная длина (мкм), нужна в изотоническом режиме

    Возвращает:
        dy/dt — numpy array той же формы что y
    """
    if l0 is None:
        l0 = y[_I_L2] + y[_I_L3]  # fallback: текущая длина как референсная

    ca = _compute_calcium(y, p)
    elec = _compute_electrical(y, p, t, ca)
    mech = _compute_mechanical(y, p, F_afterload, l0)

    dy = np.empty(_N_STATES)
    dy[_I_D] = elec["dd"]
    dy[_I_F2] = elec["df2"]
    dy[_I_FCASS] = elec["dfCass"]
    dy[_I_F] = elec["df"]
    dy[_I_CA_SR] = 0.0  # dCa_SR = 0 (переменная сохранена для совместимости)
    dy[_I_CA_I] = elec["dCa_i"]
    dy[_I_CA_SS] = elec["dCa_ss"]
    dy[_I_P_IUP] = ca["dp_iup"]
    dy[_I_H] = elec["dh"]
    dy[_I_J] = elec["dj"]
    dy[_I_M] = elec["dm"]
    dy[_I_V] = elec["dV"]
    dy[_I_K_I] = elec["dK_i"]
    dy[_I_XR1] = elec["dXr1"]
    dy[_I_XR2] = elec["dXr2"]
    dy[_I_XS] = elec["dXs"]
    dy[_I_NA_I] = elec["dNa_i"]
    dy[_I_R_gate] = elec["dr"]
    dy[_I_S] = elec["ds"]
    dy[_I_V_MECH] = mech["dv"]
    dy[_I_W_MECH] = mech["dw"]
    dy[_I_N] = mech["dN"]
    dy[_I_A] = elec["dA"]
    dy[_I_L1] = mech["dl_1"]
    dy[_I_L2] = mech["dl_2"]
    dy[_I_L3] = mech["dl_3"]
    dy[_I_R_RYR] = ca["dR"]
    dy[_I_O] = ca["dO"]
    dy[_I_I_RYR] = ca["dI"]
    dy[_I_RI] = ca["dRI"]
    dy[_I_CAMKT] = ca["dCaMKt"]
    dy[_I_CA_NSR] = ca["dCa_nSR"]
    dy[_I_CA_JSR] = ca["dCa_jSR"]

    return dy


def calculate_outputs(
    t_arr: np.ndarray,
    y_arr: np.ndarray,
    p: TNNPMParams,
    F_afterload: float = 0.0,
    l0: float | None = None,
) -> dict:
    """Вычислить токи и силы по решённой траектории.

    Аргументы:
        t_arr  — массив времён (N,)
        y_arr  — матрица состояний (N, 33)
        p      — параметры модели

    Возвращает dict с массивами длины N:
        currents: i_Na, i_CaL, i_NaCa, i_NaK, i_K1, i_Kr, i_Ks,
                  i_K_ATP, i_to, i_rel, i_up, i_leak, i_xfer, i_b_Ca, i_p_Ca
        forces:   F_CE, F_SE, F_PE, F_VS1, F_VS2, F_XSE
    """
    N = len(t_arr)

    keys_c = [
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
    keys_f = ["F_CE", "F_SE", "F_PE", "F_VS1", "F_VS2", "F_XSE"]

    out = {k: np.empty(N) for k in keys_c + keys_f}

    for i in range(N):
        y = y_arr[i]
        _l0 = l0 if l0 is not None else y[_I_L2] + y[_I_L3]

        ca = _compute_calcium(y, p)
        elec = _compute_electrical(y, p, t_arr[i], ca)
        mech = _compute_mechanical(y, p, F_afterload, _l0)

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
            "i_b_Ca",
            "i_p_Ca",
        ]:
            out[k][i] = elec[k]
        out["i_rel"][i] = ca["i_rel"]
        out["i_up"][i] = ca["i_up"]
        out["i_leak"][i] = ca["i_leak"]
        out["i_xfer"][i] = ca["i_xfer"]

        for k in ["F_CE", "F_SE", "F_PE", "F_VS1", "F_VS2", "F_XSE"]:
            out[k][i] = mech[k]

    return out


# ---------------------------------------------------------------------------
# Инициализация механики
# ---------------------------------------------------------------------------


def _N0(l0: float, p: TNNPMParams) -> float:
    return (p.r0 - p.beta_2 * (exp(p.alpha_2 * l0) - 1.0)) / p.llambda


def _L(l0: float, p: TNNPMParams) -> float:
    return (
        l0
        + (
            log(p.beta_1)
            - log(p.r0 + p.beta_1 - p.beta_2 * (exp(p.alpha_2 * l0) - 1.0))
        )
        / p.alpha_1
    )


def _fi(l_1: float, p: TNNPMParams) -> float:
    A_ref = 6.31929074e-04  # начальное CaTnC из стационарного состояния
    return _k_p_v(0.0, p) * _M(A_ref, p) * _n_1(l_1, p) * _L_oz(l_1, p) * (
        1.0 - _N0(l_1, p)
    ) - _k_m_v(0.0, p) * _N0(l_1, p)


def make_init_state(p: TNNPMParams, base: InitialState | None = None) -> InitialState:
    """Вычислить самосогласованные начальные условия для механики.

    Заменяет delenie() + calculate_init_conditions() из оригинала.
    Использует scipy.optimize.brentq вместо ручной бисекции.

    Аргументы:
        p    — параметры модели
        base — если передан, обновляются только механические переменные,
               остальные берутся из base. Если None — используется InitialState().

    Возвращает новый InitialState с пересчитанными l_1, l_2, l_3, N, v_mech, w_mech.
    """
    if base is None:
        base = InitialState()

    l200 = log((p.r0 + p.beta_2) / p.beta_2) / p.alpha_2
    a, b = 0.9 * l200, l200

    l_2 = brentq(_fi, a, b, args=(p,), xtol=1e-10, rtol=1e-10)
    l_1 = _L(l_2, p)
    N = _N0(l_2, p)
    l_3 = log((p.r0 + p.beta_3) / p.beta_3) / p.alpha_3
    l0 = l_2 + l_3

    return (
        base.replace(
            v_mech=0.0,
            w_mech=0.0,
            N=N,
            l_1=l_1,
            l_2=l_2,
            l_3=l_3,
        ),
        l0,
    )
