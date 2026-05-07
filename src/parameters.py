"""
parameters.py
=============
Параметры модели TNNPM разделены на два независимых dataclass:

  TNNPMParams  — физические константы и проводимости модели.
                 Не меняются между прогонами в рамках одного эксперимента.
                 Используются для параметрических исследований и оптимизации.

  InitialState — начальные условия (фазовые переменные).
                 Загружаются из файла или задаются вручную перед каждым прогоном.

Оба класса frozen=True:
  - значения нельзя случайно изменить после создания
  - можно использовать в качестве ключей словаря / хранить в множестве
  - безопасны при параллельных прогонах (нет общего изменяемого состояния)

Изменение параметров для sweep / оптимизации:
  new_params = dataclasses.replace(base_params, g_CaL=1.2e-4, g_Na=16.0)

Сериализация в dict (для сохранения в HDF5):
  d = dataclasses.asdict(params)

Загрузка из dict (из HDF5 или YAML):
  params = TNNPMParams(**d)
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Вспомогательная функция
# ---------------------------------------------------------------------------


def _replace(obj, **kwargs):
    """Удобный алиас для dataclasses.replace().

    Пример:
        new_params = _replace(base_params, g_CaL=1.2e-4)
    """
    return dataclasses.replace(obj, **kwargs)


# ---------------------------------------------------------------------------
# TNNPMParams — физические параметры модели
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TNNPMParams:
    """Физические параметры модели TNNPM.

    Сгруппированы по подсистемам:
      1. Параметры стимуляции
      2. Параметры клетки (константы, объёмы)
      3. Ионные токи — электрическая часть
      4. Кальциевая динамика (SR, буферы)
      5. CaMK / Semin
      6. Ишемические параметры (K-ATP)
      7. Механическая часть (ЭКБ-модель)

    Все значения по умолчанию соответствуют стандартной конфигурации
    из оригинального calculate_parameters.py.
    """

    # ------------------------------------------------------------------
    # 1. Стимуляция
    # ------------------------------------------------------------------
    stim_start: float = 10.0  # мс, начало стимула в периоде
    stim_period: float = 1000.0  # мс, период стимуляции (1 Гц)
    stim_duration: float = 1.0  # мс, длительность стимула
    stim_amplitude: float = 52.0  # пА/пФ, амплитуда стимула

    # ------------------------------------------------------------------
    # 2. Параметры клетки
    # ------------------------------------------------------------------
    T: float = 310.0  # К, температура
    F: float = 96485.3415  # Кл/ммоль, постоянная Фарадея
    R: float = 8314.472  # Дж/(моль·К), газовая постоянная
    Cm: float = 0.185  # мкФ, ёмкость мембраны

    # Внеклеточные концентрации (мМ)
    Ca_o: float = 2.0
    Na_o: float = 140.0
    K_o: float = 5.4

    # Объёмы компартментов (мкм³)
    V_c: float = 0.016404
    V_sr: float = 0.001094
    V_ss: float = 5.47e-05
    V_nSR: float = 0.00100648
    V_jSR: float = 0.00008752

    # ------------------------------------------------------------------
    # 3. Ионные токи — проводимости и параметры
    # ------------------------------------------------------------------

    # i_Na — быстрый натриевый ток
    g_Na: float = 14.838  # нСм/пФ

    # i_CaL — L-тип кальциевый ток
    g_CaL: float = 5.0e-5  # л/(Ф·с)

    # i_to — переходящий наружный ток
    g_to: float = 0.735  # нСм/пФ

    # i_Kr — быстрый калиевый выпрямляющий ток
    g_Kr: float = 0.153  # нСм/пФ

    # i_Ks — медленный калиевый ток
    g_Ks: float = 0.392  # нСм/пФ
    P_kna: float = 0.03  # безразм.

    # i_K1 — входящий выпрямляющий калиевый ток
    g_K1: float = 5.405  # нСм/пФ

    # i_NaK — насос Na-K
    P_NaK: float = 2.724  # пА/пФ
    K_mk: float = 1.0  # мМ
    K_mNa: float = 40.0  # мМ

    # i_NaCa — обменник Na-Ca
    K_NaCa: float = 5000.0  # пА/пФ
    gamma: float = 0.35  # безразм.
    alpha_NaCa: float = 1.0  # безразм. (alpha в оригинале)
    K_sat: float = 0.1  # безразм.
    Km_Ca: float = 1.38  # мМ
    Km_Nai: float = 87.5  # мМ

    # i_b_Ca — фоновый кальциевый ток
    g_bCa: float = 0.000592  # нСм/пФ

    # i_p_Ca — кальциевый насос
    g_pCa: float = 0.2476  # пА/пФ
    K_pCa: float = 0.0005  # мМ

    # i_b_Na — фоновый натриевый ток
    g_bna: float = 0.00029  # нСм/пФ

    # i_p_K — калиевый насос
    g_pK: float = 0.0146  # нСм/пФ

    # ------------------------------------------------------------------
    # 4. Кальциевая динамика
    # ------------------------------------------------------------------

    # Буферы (мМ)
    Buf_c: float = 0.11
    K_buf_c: float = 0.00085
    Buf_sr: float = 10.0
    K_buf_sr: float = 0.3
    Buf_ss: float = 0.4
    K_buf_ss: float = 0.00025
    Buf_jsr: float = 10.0
    K_buf_jsr: float = 0.3

    # Насос SERCA (i_up)
    Vmax_up: float = 0.00058  # мМ/мс
    K_up: float = 0.00025  # мМ

    # Утечка SR (i_leak)
    V_leak: float = 0.00036  # мс⁻¹

    # Выброс Ca (i_rel, Shanon RyR)
    V_rel: float = 2.5  # мс⁻¹
    k_o_Ca: float = 2.1  # мМ⁻²·мс⁻¹
    k_i_Ca: float = 0.025  # мМ⁻¹·мс⁻¹
    k_om: float = 0.06  # мс⁻¹
    k_im: float = 0.005  # мс⁻¹
    max_sr: float = 2.5
    min_sr: float = 1.0
    EC: float = 1.5  # мМ

    # Перенос Ca (i_xfer)
    V_xfer: float = 0.00456  # мс⁻¹

    # Связывание Ca с TnC (дA)
    A_tot: float = 0.07  # мМ, общий TnC
    a_on: float = 36.0  # с⁻¹
    a_off: float = 0.19  # с⁻¹
    k_A: float = 28.0  # мМ⁻¹

    # Semin: фосфорилирование p_iup
    k_sm_p: float = 1000.0  # kp
    K_lrg_p: float = 0.000325  # KP, мМ

    # ------------------------------------------------------------------
    # 5. CaMK (кальмодулин-зависимая протеинкиназа II)
    # ------------------------------------------------------------------
    CaMKo: float = 0.05  # мМ, общая концентрация CaMK
    KmCaM: float = 0.0015  # мМ, константа диссоциации кальмодулина
    KmCaMK: float = 0.15  # мМ, константа полуактивации CaMK
    aCaMK: float = 0.05  # мс⁻¹, скорость автофосфорилирования
    bCaMK: float = 0.00068  # мс⁻¹, скорость дефосфорилирования

    # ------------------------------------------------------------------
    # 6. Ишемические параметры (K-ATP канал)
    # ------------------------------------------------------------------
    g_K_ATP: float = 1.59294  # gKATP
    ATPi: float = 6.8  # мМ
    KmATP: float = 0.0976  # мМ

    # ------------------------------------------------------------------
    # 7. Механическая часть (ЭКБ-модель)
    # ------------------------------------------------------------------

    # Структурные параметры
    llambda: float = 450.0  # мН (lambda)
    r0: float = 2.55248904424517  # мН

    # Упругие элементы (экспоненциальные пружины)
    alpha_1: float = 14.6  # мкм⁻¹  (SE: последовательный)
    beta_1: float = 4.2  # мН
    alpha_2: float = 14.6  # мкм⁻¹  (PE: параллельный)
    beta_2: float = 0.009  # мН
    alpha_3: float = 55.0  # мкм⁻¹  (XSE: внешний)
    beta_3: float = 0.11  # мН

    # Вязкие элементы (CE)
    alpha_vp_l: float = 16.0  # мкм⁻¹
    beta_vp_l: float = 0.1  # мН·с/мкм
    alpha_vp_s: float = 16.0
    beta_vp_s: float = 10.0
    alpha_vs_l: float = 46.0
    beta_vs_l: float = 20.0
    alpha_vs_s: float = 39.0
    beta_vs_s: float = 60.0

    # Кинетика мостиков
    chi_0: float = 2.1  # каппа0
    chi_1: float = 0.55  # каппа1
    chi_2: float = 0.0  # каппа2
    v_max: float = 0.0055  # мкм/мс, макс. скорость укорочения
    m_0: float = 0.9  # начальная вероятность прикрепления

    # Параметры q(v)
    q_1: float = 0.0173
    q_2: float = 0.259
    q_3: float = 0.0173
    q_4: float = 0.015
    x_st: float = 0.964285
    alpha_Q: float = 10.0
    beta_Q: float = 5.0

    # G*(v)
    a: float = 0.25
    v_1: float = 0.1
    alpha_G: float = 1.0
    alpha_P: float = 4.0
    d_h: float = 0.5

    # M(A) — зависимость от Ca через TnC
    mu: float = 3.3
    k_mu: float = 0.6

    # n_1(l_1) — длиннозависимая активация
    g_1: float = 0.6
    g_2: float = 0.52
    n1_A: float = 0.5
    n1_B: float = 55.0
    n1_C: float = 1.0
    n1_K: float = 1.0
    n1_Q: float = 0.835
    n1_nu: float = 5.0

    # L_oz(l_1) — Frank-Starling геометрия
    s_0: float = 1.14
    s046: float = 0.46
    s055: float = 0.55

    # pi_N_A — модуляция pi_min
    s_c: float = 1.0
    pi_min: float = 0.02

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    def replace(self, **kwargs) -> TNNPMParams:
        """Создать копию с изменёнными полями.

        Пример (sweep по g_CaL):
            for scale in [0.8, 1.0, 1.2]:
                p = base_params.replace(g_CaL=base_params.g_CaL * scale)
        """
        return dataclasses.replace(self, **kwargs)

    def to_dict(self) -> dict:
        """Сериализация в dict (для HDF5 / JSON)."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> TNNPMParams:
        """Десериализация из dict.

        Лишние ключи (например, устаревшие параметры из старых файлов)
        игнорируются, чтобы не ломать обратную совместимость.
        """
        known = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# InitialState — начальные условия системы ОДУ
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InitialState:
    """Начальные условия для системы ОДУ модели TNNPM.

    Порядок полей СТРОГО соответствует индексам в векторе состояния y:
      [0]  d        [1]  f2       [2]  fCass    [3]  f
      [4]  Ca_SR    [5]  Ca_i     [6]  Ca_ss    [7]  p_iup
      [8]  h        [9]  j        [10] m        [11] V
      [12] K_i      [13] Xr1      [14] Xr2      [15] Xs
      [16] Na_i     [17] r        [18] s
      [19] v        [20] w        [21] N        [22] A
      [23] l_1      [24] l_2      [25] l_3
      [26] R        [27] O        [28] I        [29] RI
      [30] CaMKt    [31] Ca_nSR   [32] Ca_jSR

    Значения по умолчанию — стационарное состояние при 1 Гц.

    Загрузка из результатов предыдущего прогона:
        state0 = InitialState.from_result(results_load['variables'])

    Изменение отдельных переменных:
        state0 = base_state.replace(V=-85.0, Ca_i=5e-5)
    """

    # Электрическая часть — ворота TNNP
    d: float = 3.07278916e-05
    f2: float = 9.99476773e-01
    fCass: float = 9.99970647e-01
    f: float = 9.82107417e-01
    Ca_SR: float = 9.28218670e-01  # устаревшая переменная, dCa_SR=0 (см. model.py)
    Ca_i: float = 4.34295055e-05
    Ca_ss: float = 1.57457522e-04
    p_iup: float = 4.88823882e-01
    h: float = 7.63521043e-01
    j: float = 7.62863363e-01
    m: float = 1.47900595e-03
    V: float = -8.59273503e01
    K_i: float = 1.35877968e02
    Xr1: float = 1.92071800e-04
    Xr2: float = 4.78422683e-01
    Xs: float = 3.09815853e-03
    Na_i: float = 1.02371819e01
    r: float = 2.15152212e-08
    s: float = 9.99998122e-01

    # Механическая часть (ЭКБ)
    v_mech: float = (
        0.0  # скорость CE (переименовано: v → v_mech во избежание конфликта с V)
    )
    w_mech: float = 0.0  # скорость PE (переименовано: w → w_mech)
    N: float = 1.34966357e-06  # доля прикреплённых мостиков
    A: float = 6.31929074e-04  # CaTnC (связанный Ca с TnC)
    l_1: float = 3.86134451e-01  # длина CE
    l_2: float = 3.86910662e-01  # длина PE
    l_3: float = 5.80515391e-02  # длина XSE

    # RyR (Shanon)
    R: float = 9.88267582e-01
    O: float = 4.23457956e-07
    I: float = 5.02698275e-09
    RI: float = 1.17319890e-02

    # CaMK
    CaMKt: float = 0.0110752904836162

    # SR (разделённый на nSR и jSR)
    Ca_nSR: float = 4.28218670e-01
    Ca_jSR: float = 4.88218670e-01

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    def to_array(self) -> list:
        """Преобразовать в список для передачи в солвер.

        Порядок строго соответствует индексам вектора состояния.
        """
        return [
            self.d,
            self.f2,
            self.fCass,
            self.f,
            self.Ca_SR,
            self.Ca_i,
            self.Ca_ss,
            self.p_iup,
            self.h,
            self.j,
            self.m,
            self.V,
            self.K_i,
            self.Xr1,
            self.Xr2,
            self.Xs,
            self.Na_i,
            self.r,
            self.s,
            self.v_mech,
            self.w_mech,
            self.N,
            self.A,
            self.l_1,
            self.l_2,
            self.l_3,
            self.R,
            self.O,
            self.I,
            self.RI,
            self.CaMKt,
            self.Ca_nSR,
            self.Ca_jSR,
        ]

    @classmethod
    def from_array(cls, y: list) -> InitialState:
        """Создать из вектора состояния (обратная операция к to_array)."""
        return cls(
            d=y[0],
            f2=y[1],
            fCass=y[2],
            f=y[3],
            Ca_SR=y[4],
            Ca_i=y[5],
            Ca_ss=y[6],
            p_iup=y[7],
            h=y[8],
            j=y[9],
            m=y[10],
            V=y[11],
            K_i=y[12],
            Xr1=y[13],
            Xr2=y[14],
            Xs=y[15],
            Na_i=y[16],
            r=y[17],
            s=y[18],
            v_mech=y[19],
            w_mech=y[20],
            N=y[21],
            A=y[22],
            l_1=y[23],
            l_2=y[24],
            l_3=y[25],
            R=y[26],
            O=y[27],
            I=y[28],
            RI=y[29],
            CaMKt=y[30],
            Ca_nSR=y[31],
            Ca_jSR=y[32],
        )

    @classmethod
    def from_result(cls, variables: dict) -> InitialState:
        """Загрузить начальные условия из последней точки предыдущего прогона.

        Аргумент variables — словарь массивов из SimulationResult.variables
        (или напрямую из loadh5).

        Пример:
            result = loadh5('data/steady_state.h5')
            state0 = InitialState.from_result(result['variables'])
        """

        def last(key):
            return float(variables[key][-1])

        return cls(
            d=last("d"),
            f2=last("f2"),
            fCass=last("fCass"),
            f=last("f"),
            Ca_SR=last("Ca_SR"),
            Ca_i=last("Ca_i"),
            Ca_ss=last("Ca_ss"),
            p_iup=last("p_iup"),
            h=last("h"),
            j=last("j"),
            m=last("m"),
            V=last("V"),
            K_i=last("K_i"),
            Xr1=last("Xr1"),
            Xr2=last("Xr2"),
            Xs=last("Xs"),
            Na_i=last("Na_i"),
            r=last("r"),
            s=last("s"),
            v_mech=last("v"),
            w_mech=last("w"),
            N=last("N"),
            A=last("A"),
            l_1=last("l_1"),
            l_2=last("l_2"),
            l_3=last("l_3"),
            R=last("R"),
            O=last("O"),
            I=last("I"),
            RI=last("RI"),
            CaMKt=last("CaMKt"),
            Ca_nSR=last("Ca_nSR"),
            Ca_jSR=last("Ca_jSR"),
        )

    def replace(self, **kwargs) -> InitialState:
        """Создать копию с изменёнными полями."""
        return dataclasses.replace(self, **kwargs)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> InitialState:
        known = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Конфигурация симуляции (отдельно от параметров модели)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimConfig:
    """Параметры прогона: временной диапазон и настройки солвера.

    Отделены от TNNPMParams намеренно: они не являются физическими
    параметрами модели и не должны варьироваться при параметрических
    исследованиях физики клетки.
    """

    time_start: float = 0.0  # мс
    time_stop: float = 10000.0  # мс (10 сек = 10 циклов при 1 Гц)

    # Настройки CVode
    atol: float = 1e-9
    rtol: float = 1e-9
    max_step: float = 0.5  # мс, maxh

    # Число точек вывода (None = авто по солверу)
    n_out: int | None = None

    def replace(self, **kwargs) -> SimConfig:
        return dataclasses.replace(self, **kwargs)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> SimConfig:
        known = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Готовые пресеты
# ---------------------------------------------------------------------------

#: Стандартные параметры — отправная точка для всех прогонов
DEFAULT_PARAMS = TNNPMParams()

#: Начальные условия из стационарного состояния при 1 Гц
DEFAULT_STATE = InitialState()

#: Короткий прогон для тестов (1 цикл)
TEST_CONFIG = SimConfig(time_stop=1000.0)

#: Стандартный прогон (500 циклов при 1 Гц)
DEFAULT_CONFIG = SimConfig(time_stop=500_000.0)
