# TNNPM — Электромеханическая модель кардиомиоцита

Реализация модели **Ten Tusscher – Niederer – Noble – Panfilov (TNNP)** с расширениями:

- механика контрактильного аппарата по модели **ЭКБ**
- RyR-каналы SR по модели **Shanon**
- **CaMK**-фосфорилирование (модуляция SERCA)
- **Semin**-модуляция насоса p_iup

Модель описывает электромеханическое сопряжение в кардиомиоците: от потенциала действия через внутриклеточный кальций до механической силы.

---

## Содержание

1. [Структура проекта](#структура-проекта)
2. [Установка](#установка)
3. [Быстрый старт](#быстрый-старт)
4. [CLI — запуск из командной строки](#cli)
5. [Python API](#python-api)
6. [Конфигурационные файлы YAML](#конфигурационные-файлы-yaml)
7. [Описание модулей](#описание-модулей)
8. [Переменные состояния](#переменные-состояния)
9. [Параметры модели](#параметры-модели)
10. [Форматы файлов](#форматы-файлов)
11. [Тесты](#тесты)
12. [Архитектурные решения](#архитектурные-решения)

---

## Структура проекта

```
tnnpm/
├── src/
│   ├── parameters.py       # TNNPMParams, InitialState, SimConfig
│   ├── model.py            # rhs(), calculate_outputs(), make_init_state()
│   ├── solver.py           # integrate() — обёртка CVode (Assimulo)
│   ├── experiment.py       # SimulationResult, run_single(), run_sweep()
│   ├── simulation_io.py    # save/load HDF5 (single, batch, legacy, state)
│   └── plots.py            # визуализация результатов
│
├── tests/
│   ├── conftest.py         # мок Assimulo на уровне sys.modules
│   ├── test_model.py       # физика модели, ворота, инварианты
│   ├── test_experiment.py  # run_single/sweep, chain-run, SimulationResult
│   ├── test_io_plots.py    # HDF5 roundtrip, batch, legacy, plot-функции
│   └── test_cli.py         # CLI: парсинг, sweep/single маршрутизация
│
├── configs/
│   ├── default.yaml        # 1 Гц, 500 циклов, изометрия
│   └── 2Hz_sweep_gCaL.yaml # sweep по g_CaL при 2 Гц
│
├── scripts/
│   └── run.py              # CLI точка входа
│
├── data/                   # результаты симуляций (HDF5, создаётся автоматически)
├── reports/figures/        # графики (создаётся автоматически)
├── environment.yml         # conda-окружение
└── pyproject.toml          # метаданные пакета
```

---

## Установка

### Требования

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) или Anaconda
- Linux / macOS (Assimulo требует компилятор C)

### Создание окружения

```bash
# Клонировать репозиторий
git clone <repo-url>
cd tnnpm

# Создать conda-окружение со всеми зависимостями (~5 минут)
conda env create -f environment.yml

# Активировать
conda activate tnnpm
```

Команда `conda env create` автоматически:

- устанавливает Assimulo с Sundials из канала `conda-forge`
- устанавливает numpy, scipy, h5py, matplotlib, joblib, loguru, pyyaml
- запускает `pip install -e .` для установки самого пакета в editable-режиме

### Проверка установки

```bash
pytest           # должно пройти 181 тест
tnnpm-run --help # CLI доступен глобально
```

### Обновление окружения

```bash
# Если environment.yml изменился
conda env update -f environment.yml --prune
```

---

## Быстрый старт

```bash
conda activate tnnpm
cd /home/akursanov/models/tnnpm

# Одиночный прогон: 1 Гц, 500 циклов (~10 мин)
python scripts/run.py --config configs/default.yaml

# Результат сохраняется в data/default.h5
# Посмотреть результат:
python scripts/run.py --config configs/default.yaml --plot --plot-last 3000
```

---

## CLI

Все симуляции можно запускать через `scripts/run.py` без написания Python-кода.

### Синтаксис

```
python scripts/run.py [опции]
```

### Опции

| Опция | Короткая | Описание |
|---|---|---|
| `--config PATH` | `-c` | YAML-конфиг. Если не указан — используются defaults |
| `--output PATH` | `-o` | Куда сохранить результат (.h5). По умолчанию: `data/<config>.h5` |
| `--warmup PATH` | `-w` | HDF5-файл с начальными условиями из предыдущего прогона |
| `--set KEY=VAL` | `-s` | Переопределить параметр прямо из командной строки |
| `--jobs N` | `-j` | Число параллельных процессов для sweep (по умолчанию: 1) |
| `--plot` | | Построить графики после прогона |
| `--plot-last MS` | | Показывать только последние N мс на графике |
| `--no-save` | | Не записывать результат на диск (для отладки) |

### Примеры

```bash
# Одиночный прогон с параметрами по умолчанию
python scripts/run.py --config configs/default.yaml

# Сохранить в конкретный файл
python scripts/run.py --config configs/default.yaml \
    --output data/1Hz_500c_baseline.h5

# Параметрический sweep: 3 значения g_CaL × 2 значения stim_period = 6 прогонов
# Запускается параллельно на 4 ядрах
python scripts/run.py --config configs/2Hz_sweep_gCaL.yaml --jobs 4

# Переопределить параметры без редактирования конфига
python scripts/run.py --config configs/default.yaml \
    --set g_Na=16.0 \
    --set ATPi=3.0 \
    --set stim_period=500

# Цепочка прогонов: выйти на стационар при 1 Гц, затем переключить на 2 Гц
python scripts/run.py --config configs/default.yaml \
    --output data/warmup_1Hz.h5

python scripts/run.py --config configs/default.yaml \
    --set stim_period=500 \
    --warmup data/warmup_1Hz.h5 \
    --output data/2Hz_from_1Hz.h5

# Прогон с графиком (только последние 3 секунды)
python scripts/run.py --config configs/default.yaml \
    --plot --plot-last 3000

# Отладка: запустить без сохранения
python scripts/run.py --config configs/default.yaml --no-save
```

### Автоматическое определение режима

CLI автоматически определяет режим по YAML-конфигу:

- **Одиночный прогон** — если все параметры в `params` скаляры
- **Sweep** — если хотя бы один параметр задан списком

```yaml
# Одиночный прогон
params:
  stim_period: 1000.0

# Sweep (3 прогона)
params:
  stim_period: 1000.0
  g_CaL: [3.0e-5, 5.0e-5, 7.0e-5]

# Sweep 2D (2 × 3 = 6 прогонов)
params:
  g_CaL: [3.0e-5, 5.0e-5, 7.0e-5]
  stim_period: [500.0, 1000.0]
```

---

## Python API

### Минимальный пример

```python
from parameters import DEFAULT_PARAMS, SimConfig
from model import make_init_state
from experiment import run_single
from simulation_io import save
from plots import plot_overview

# Начальные условия
params = DEFAULT_PARAMS
state0, l0 = make_init_state(params)

# Прогон: 10 секунд при 1 Гц
config = SimConfig(time_stop=10_000.0)
result = run_single(params, state0, config)

# Результат
print(f"Точек: {len(result.time)}")
print(f"max Ca_i = {result.Ca_i.max()*1e3:.2f} мкМ")
print(f"max F_XSE = {result.F_XSE.max():.3f} мН")

# График и сохранение
fig, axes = plot_overview(result, last_n_ms=3000.0)
fig.savefig("reports/figures/overview.png", dpi=150)
save(result, "data/result.h5")
```

### Работа с параметрами

`TNNPMParams` — неизменяемый (`frozen=True`) dataclass. Изменение создаёт новый объект:

```python
from parameters import DEFAULT_PARAMS

# Создать вариант с другим g_CaL
p2 = DEFAULT_PARAMS.replace(g_CaL=7e-5)

# Несколько параметров сразу
p3 = DEFAULT_PARAMS.replace(g_CaL=7e-5, g_Na=16.0, stim_period=500.0)

# Оригинал не изменился
assert DEFAULT_PARAMS.g_CaL == 5e-5

# Сериализация в dict (для логирования или сохранения)
d = p3.to_dict()

# Загрузка из dict (устаревшие ключи игнорируются)
from parameters import TNNPMParams
p4 = TNNPMParams.from_dict(d)
```

### Цепочка прогонов

Типичная схема: warmup на 1 Гц → переключение на 2 Гц:

```python
from parameters import DEFAULT_PARAMS, SimConfig
from model import make_init_state
from experiment import run_single
from simulation_io import save, load_state

params_1hz = DEFAULT_PARAMS                            # 1 Гц
params_2hz = DEFAULT_PARAMS.replace(stim_period=500.0) # 2 Гц

config_warmup = SimConfig(time_stop=500_000.0)  # 500 циклов
config_main   = SimConfig(time_stop=500_000.0)

# Шаг 1: выйти на стационар при 1 Гц
state0, l0 = make_init_state(params_1hz)
r_warmup = run_single(params_1hz, state0, config_warmup)
save(r_warmup, "data/warmup_1Hz.h5")

# Шаг 2: продолжить при 2 Гц с последней точки warmup
state_after_warmup = r_warmup.last_state()
r_main = run_single(params_2hz, state_after_warmup, config_main,
                    recompute_init_mechanics=True)
save(r_main, "data/2Hz_from_1Hz.h5")

# Или загрузить warmup из файла (не держать в памяти)
state0_next = load_state("data/warmup_1Hz.h5")
r_main2 = run_single(params_2hz, state0_next, config_main,
                     recompute_init_mechanics=True)
```

### Параметрический sweep

```python
from parameters import DEFAULT_PARAMS, SimConfig
from model import make_init_state
from experiment import run_sweep, sweep_combinations
from simulation_io import save_batch
from plots import plot_sweep, plot_sweep_peak

params = DEFAULT_PARAMS
state0, l0 = make_init_state(params)
config = SimConfig(time_stop=100_000.0)  # 100 циклов на прогон

# Предпросмотр комбинаций без запуска
combos = sweep_combinations({'g_CaL': [3e-5, 5e-5, 7e-5], 'stim_period': [500., 1000.]})
print(f"Будет {len(combos)} прогонов:")
for c in combos:
    print(f"  {c}")

# Запуск sweep (6 прогонов, 4 параллельных процесса)
results = run_sweep(
    sweep={'g_CaL': [3e-5, 5e-5, 7e-5], 'stim_period': [500., 1000.]},
    base_params=params,
    state0=state0,
    config=config,
    n_jobs=4,
)

# Сохранить всё в один файл
save_batch(results, "data/sweep_gCaL_x_freq.h5")

# Наложенные кривые Ca_i для каждого значения g_CaL
fig, ax = plot_sweep(results, variable='Ca_i', sweep_param='g_CaL',
                     last_n_ms=2000.0)
fig.savefig("reports/figures/sweep_Ca_i.png", dpi=150)

# График «пиковая сила vs g_CaL»
fig, ax = plot_sweep_peak(results, variable='F_XSE', sweep_param='g_CaL',
                          from_forces=True, stat='max')
fig.savefig("reports/figures/sweep_peak_force.png", dpi=150)
```

### Интеграция с оптимизаторами

`rhs()` — чистая функция без глобального состояния, безопасная для параллельного запуска. Это позволяет использовать любой оптимизатор напрямую:

```python
from scipy.optimize import minimize
from parameters import DEFAULT_PARAMS, SimConfig
from model import make_init_state
from experiment import run_single

params_base = DEFAULT_PARAMS
state0, l0  = make_init_state(params_base)
config = SimConfig(time_stop=20_000.0)  # 20 циклов для быстрой оценки

def objective(x):
    """Минимизировать разницу между пиковой силой и целевым значением."""
    p = params_base.replace(g_CaL=x[0], g_Na=x[1])
    r = run_single(p, state0, config, l0=l0)
    peak_force = r.forces['F_XSE'].max()
    target = 0.15  # мН
    return (peak_force - target) ** 2

result = minimize(objective, x0=[5e-5, 14.838],
                  bounds=[(1e-5, 1e-4), (5.0, 30.0)],
                  method='Nelder-Mead')
print(f"g_CaL = {result.x[0]:.3e}, g_Na = {result.x[1]:.3f}")
```

### Загрузка и анализ результатов

```python
from simulation_io import load, load_batch, load_legacy
from parameters import InitialState
import numpy as np

# Загрузить одиночный прогон
result = load("data/2Hz_from_1Hz.h5")

# Доступ к данным
t     = result.time               # мс, shape (N,)
V     = result.V                  # мВ, то же что result.variables['V']
Ca_i  = result.Ca_i              # мМ
F_XSE = result.F_XSE             # мН
i_CaL = result.currents['i_CaL'] # пА/пФ

# Параметры и конфиг сохранены вместе с данными
print(f"Частота: {1000/result.params.stim_period:.1f} Гц")
print(f"Время счёта: {result.meta['wall_time_s']:.1f} с")

# Анализ последнего установившегося цикла
last_cycle_start = result.time[-1] - result.params.stim_period
mask = result.time >= last_cycle_start
V_cycle = V[mask]
print(f"Пиковый потенциал: {V_cycle.max():.1f} мВ")
print(f"Потенциал покоя:   {V_cycle.min():.1f} мВ")

# Загрузить sweep
results    = load_batch("data/sweep_gCaL_x_freq.h5")
peak_forces = [r.forces['F_XSE'].max() for r in results]
g_CaL_vals  = [r.params.g_CaL for r in results]

# Загрузить файл в старом формате (от оригинального saveh5)
legacy = load_legacy("data/resultsST.h5")
state0 = InitialState.from_result(legacy['variables'])
```

### Визуализация

```python
import matplotlib.pyplot as plt
from simulation_io import load
from plots import (
    plot_voltage, plot_calcium, plot_force,
    plot_currents, plot_sr_calcium, plot_overview,
    plot_sweep, plot_sweep_peak,
)

result = load("data/result.h5")

# Все функции возвращают (fig, ax) и не вызывают plt.show() — вы решаете что делать

# Сводный график (V + Ca_i + F_XSE)
fig, axes = plot_overview(result, last_n_ms=3000.0, title="Baseline 1 Hz")
fig.savefig("reports/figures/overview.png", dpi=150)
plt.close(fig)

# Отдельные переменные
fig, ax = plot_voltage(result, last_n_ms=2000.0)
fig, ax = plot_calcium(result, last_n_ms=2000.0)
fig, ax = plot_force(result, component='F_XSE', last_n_ms=2000.0)
fig, ax = plot_sr_calcium(result, last_n_ms=2000.0)  # Ca_nSR и Ca_jSR

# Набор токов на отдельных подграфиках
fig, axes = plot_currents(result,
                          currents=['i_Na', 'i_CaL', 'i_NaCa', 'i_NaK'],
                          last_n_ms=2000.0)

# Вставить в существующий subplot
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
plot_voltage(result,   last_n_ms=2000.0, ax=axes[0, 0])
plot_calcium(result,   last_n_ms=2000.0, ax=axes[0, 1])
plot_force(result,     last_n_ms=2000.0, ax=axes[1, 0])
plot_sr_calcium(result, last_n_ms=2000.0, ax=axes[1, 1])
fig.tight_layout()
```

---

## Конфигурационные файлы YAML

Конфиги управляют параметрами модели и солвера без изменения кода.

### Структура

```yaml
# Параметры модели (поля TNNPMParams)
params:
  stim_period:    1000.0   # мс (1 Гц)
  stim_start:     10.0
  stim_duration:  1.0
  stim_amplitude: 52.0

  g_Na:    14.838
  g_CaL:   5.0e-5
  Vmax_up: 0.00058
  ATPi:    6.8             # мМ (норма); уменьшить для ишемии

# Параметры прогона (поля SimConfig)
config:
  time_start: 0.0
  time_stop:  500000.0     # мс
  atol:       1.0e-9
  rtol:       1.0e-9
  max_step:   0.5          # мс, максимальный шаг CVode
  n_out:      ~            # ~ = None: CVode выбирает точки сам

# Режим нагрузки
mechanics:
  F_afterload: 0.0         # мН (0 = изометрия)

# Начальные условия из файла (опционально)
# warmup_file: data/warmup_1Hz.h5
```

### Sweep в конфиге

Список значений вместо скаляра автоматически запускает параметрический sweep. Перебираются все комбинации (декартово произведение):

```yaml
params:
  stim_period: 500.0                   # скаляр — не варьируется
  g_CaL: [3.0e-5, 5.0e-5, 7.0e-5]    # 3 значения
  g_Kr:  [0.10, 0.15]                  # 2 значения
  # итого: 3 × 2 = 6 прогонов
```

### Готовые конфиги

| Файл | Описание |
|---|---|
| `configs/default.yaml` | 1 Гц, 500 циклов, изометрия, стандартные параметры |
| `configs/2Hz_sweep_gCaL.yaml` | 2 Гц, sweep по g_CaL: [3e-5, 5e-5, 7e-5] |

---

## Описание модулей

### `parameters.py`

Три неизменяемых (`frozen=True`) dataclass:

**`TNNPMParams`** — все физические параметры модели (~80 полей). Сгруппированы по подсистемам: стимуляция, параметры клетки, ионные токи, кальциевая динамика, CaMK, ишемические параметры, механическая часть. Методы: `replace(**kwargs)`, `to_dict()`, `from_dict(d)`.

**`InitialState`** — начальные условия (33 переменные). Порядок строго соответствует вектору состояния ОДУ. Методы: `to_array()`, `from_array(y)`, `from_result(variables)`, `replace(**kwargs)`.

**`SimConfig`** — параметры прогона: диапазон времени, допуски CVode, число точек вывода. Методы: `replace(**kwargs)`, `to_dict()`, `from_dict(d)`.

Пресеты: `DEFAULT_PARAMS`, `DEFAULT_STATE`, `DEFAULT_CONFIG`, `TEST_CONFIG`.

### `model.py`

Математическое ядро. Три публичные функции:

**`rhs(t, y, p, F_afterload, l0)`** — правая часть системы ОДУ. Чистая функция: нет изменяемого состояния, нет `self`. Вызывается солвером на каждом шаге интегрирования. Внутри использует три приватных блока: `_compute_calcium`, `_compute_electrical`, `_compute_mechanical`.

**`calculate_outputs(t_arr, y_arr, p, ...)`** — вычисляет токи и силы по сохранённой траектории. Использует те же приватные блоки что и `rhs()`, без дублирования формул. Вызывается однократно после завершения интегрирования.

**`make_init_state(p, base)`** — вычисляет самосогласованные начальные условия для механики методом `scipy.optimize.brentq`. Возвращает `(InitialState, l0)`.

### `solver.py`

Тонкая обёртка над CVode (Assimulo). Единственная функция:

**`integrate(rhs_fn, y0, config)`** — интегрирует систему ОДУ методом BDF (Backward Differentiation Formula) с итерацией Ньютона и плотным линейным решателем LAPACK. Возвращает `(t, y)`.

Настройки CVode берутся из `SimConfig`. Параллелизм организован на уровне `run_sweep` (много независимых прогонов), а не внутри одного прогона (`num_threads=1`).

### `experiment.py`

Высокоуровневый API для запуска симуляций.

**`SimulationResult`** — неизменяемый dataclass с результатом одного прогона: `time`, `variables`, `currents`, `forces`, `params`, `state0`, `config`, `meta`. Свойства-сокращения: `result.V`, `result.Ca_i`, `result.F_XSE`. Метод `last_state()` возвращает последнюю точку как `InitialState`.

**`run_single(params, state0, config, ...)`** — запускает один прогон. Если `state0=None` — автоматически вычисляет начальные условия. Аргумент `recompute_init_mechanics=True` нужен при смене механических параметров между прогонами.

**`run_sweep(sweep, base_params, ...)`** — параметрический sweep. При `n_jobs > 1` использует `joblib.Parallel`. Порядок результатов соответствует `itertools.product` варьируемых параметров.

**`sweep_combinations(sweep)`** — предпросмотр всех комбинаций без запуска симуляций.

### `simulation_io.py`

Сохранение и загрузка в формате HDF5.

| Функция | Описание |
|---|---|
| `save(result, path)` | Сохранить `SimulationResult` |
| `load(path)` | Загрузить `SimulationResult` |
| `save_batch(results, path)` | Сохранить список результатов sweep |
| `load_batch(path)` | Загрузить список результатов sweep |
| `load_state(path)` | Только последняя точка как `InitialState` (экономит память) |
| `load_legacy(path)` | Читает файлы старого формата (`saveh5` из utils.py) |
| `load_config(path)` | Загрузить `TNNPMParams` + `SimConfig` из YAML |

### `plots.py`

Функции визуализации над `SimulationResult`. Каждая создаёт независимую `Figure` и возвращает `(fig, ax)`. Принимают необязательный аргумент `ax=` для вставки в существующий subplot.

| Функция | Описание |
|---|---|
| `plot_voltage(result, ...)` | Потенциал действия V(t) |
| `plot_calcium(result, ...)` | Внутриклеточный [Ca²⁺]ᵢ (в мкМ) |
| `plot_force(result, component, ...)` | Механическая сила (любая компонента) |
| `plot_currents(result, currents, ...)` | Набор токов на подграфиках |
| `plot_sr_calcium(result, ...)` | Ca_nSR и Ca_jSR |
| `plot_overview(result, ...)` | Сводный: V + Ca_i + F_XSE |
| `plot_sweep(results, variable, sweep_param, ...)` | Наложенные кривые для sweep |
| `plot_sweep_peak(results, variable, sweep_param, stat, ...)` | Пиковое значение vs параметр |

Все функции поддерживают `last_n_ms` — показывать только последние N мс (убирает переходный процесс).

---

## Переменные состояния

Вектор состояния `y` содержит 33 переменных в фиксированном порядке:

| Индекс | Имя | Описание | Единица |
|---|---|---|---|
| 0 | `d` | Активационные ворота i_CaL | — |
| 1 | `f2` | Инактивационные ворота i_CaL (медленные) | — |
| 2 | `fCass` | Ворота i_CaL (зависят от Ca_ss) | — |
| 3 | `f` | Инактивационные ворота i_CaL | — |
| 4 | `Ca_SR` | Ca в SR (заморожена: dCa_SR=0) | мМ |
| 5 | `Ca_i` | Внутриклеточный кальций | мМ |
| 6 | `Ca_ss` | Кальций в субсарколемном пространстве | мМ |
| 7 | `p_iup` | Степень фосфорилирования SERCA (Semin) | — |
| 8 | `h` | Инактивационные ворота i_Na (быстрые) | — |
| 9 | `j` | Инактивационные ворота i_Na (медленные) | — |
| 10 | `m` | Активационные ворота i_Na | — |
| 11 | `V` | Потенциал мембраны | мВ |
| 12 | `K_i` | Внутриклеточный калий | мМ |
| 13 | `Xr1` | Активационные ворота i_Kr | — |
| 14 | `Xr2` | Инактивационные ворота i_Kr | — |
| 15 | `Xs` | Ворота i_Ks | — |
| 16 | `Na_i` | Внутриклеточный натрий | мМ |
| 17 | `r` | Активационные ворота i_to | — |
| 18 | `s` | Инактивационные ворота i_to | — |
| 19 | `v_mech` | Скорость укорочения CE | мкм/мс |
| 20 | `w_mech` | Скорость укорочения PE | мкм/мс |
| 21 | `N` | Доля прикреплённых поперечных мостиков | — |
| 22 | `A` | Ca-TnC (связанный кальций) | мМ |
| 23 | `l_1` | Длина контрактильного элемента CE | мкм |
| 24 | `l_2` | Длина параллельного элемента PE | мкм |
| 25 | `l_3` | Длина внешнего упругого элемента XSE | мкм |
| 26 | `R` | Доля RyR-каналов в состоянии R (покой) | — |
| 27 | `O` | Доля открытых RyR-каналов | — |
| 28 | `I` | Доля инактивированных RyR-каналов | — |
| 29 | `RI` | Доля RyR в состоянии RI | — |
| 30 | `CaMKt` | Доля фосфорилированной CaMK | — |
| 31 | `Ca_nSR` | Кальций в сетевом SR | мМ |
| 32 | `Ca_jSR` | Кальций в юнкциональном SR | мМ |

**Инвариант:** `R + O + I + RI = 1` (сохраняется в процессе интегрирования).

---

## Параметры модели

Основные параметры, которые чаще всего варьируют в исследованиях:

### Стимуляция

| Параметр | Default | Единица | Описание |
|---|---|---|---|
| `stim_period` | 1000.0 | мс | Период (1000 = 1 Гц, 500 = 2 Гц) |
| `stim_amplitude` | 52.0 | пА/пФ | Амплитуда стимула |
| `stim_duration` | 1.0 | мс | Длительность стимула |

### Ионные токи

| Параметр | Default | Единица | Ток |
|---|---|---|---|
| `g_Na` | 14.838 | нСм/пФ | i_Na — быстрый натриевый |
| `g_CaL` | 5.0e-5 | л/(Ф·с) | i_CaL — L-тип кальциевый |
| `g_Kr` | 0.153 | нСм/пФ | i_Kr — быстрый калиевый |
| `g_Ks` | 0.392 | нСм/пФ | i_Ks — медленный калиевый |
| `g_K1` | 5.405 | нСм/пФ | i_K1 — входящий выпрямляющий |
| `g_to` | 0.735 | нСм/пФ | i_to — переходящий наружный |
| `K_NaCa` | 5000.0 | пА/пФ | i_NaCa — обменник Na-Ca |
| `P_NaK` | 2.724 | пА/пФ | i_NaK — насос Na-K |

### Кальциевая динамика

| Параметр | Default | Единица | Описание |
|---|---|---|---|
| `Vmax_up` | 0.00058 | мМ/мс | Максимальная скорость SERCA |
| `V_rel` | 2.5 | мс⁻¹ | Скорость выброса Ca через RyR |
| `V_leak` | 0.00036 | мс⁻¹ | Утечка Ca из SR |
| `k_sm_p` | 1000.0 | — | Скорость фосфорилирования p_iup (Semin) |

### Ишемия (K-ATP канал)

| Параметр | Default | Описание |
|---|---|---|
| `ATPi` | 6.8 мМ | Норма. Уменьшить до 1–3 мМ для ишемии |
| `KmATP` | 0.0976 мМ | Константа полуактивации K-ATP |
| `g_K_ATP` | 1.593 нСм/пФ | Максимальная проводимость K-ATP |

### Механическая часть

| Параметр | Default | Описание |
|---|---|---|
| `llambda` | 450.0 мН | Жёсткость CE |
| `v_max` | 0.0055 мкм/мс | Максимальная скорость укорочения |
| `chi_1` | 0.55 | Коэффициент кинетики мостиков (каппа1) |
| `A_tot` | 0.07 мМ | Общая концентрация TnC |
| `F_afterload` | 0.0 мН | Постнагрузка (0 = изометрия) |

---

## Форматы файлов

### HDF5 (новый формат)

Файлы, созданные `save()`, содержат полную информацию о прогоне:

```
result.h5
├── time                     # shape (N,), мс
├── variables/
│   ├── V                    # shape (N,), мВ
│   ├── Ca_i                 # shape (N,), мМ
│   └── ...                  # все 33 переменные
├── currents/
│   ├── i_Na                 # shape (N,)
│   ├── i_CaL
│   └── ...                  # все 15 токов
├── forces/
│   ├── F_XSE                # shape (N,)
│   └── ...                  # все 6 сил
├── params/                  # атрибуты: все поля TNNPMParams
├── state0/                  # атрибуты: все поля InitialState
├── config/                  # атрибуты: все поля SimConfig
└── meta/                    # атрибуты: wall_time_s, n_points, ...
```

Файлы sweep (`save_batch`) содержат группы `/run_0/`, `/run_1/`, ... с той же структурой.

### HDF5 (старый формат)

Файлы от оригинального `saveh5()` читаются через `load_legacy()`:

```python
from simulation_io import load_legacy
from parameters import InitialState

d = load_legacy("data/resultsST.h5")
# d['variables']['V'], d['variables']['Ca_i'], ...

# Конвертация в InitialState для продолжения прогона
state0 = InitialState.from_result(d['variables'])
```

---

## Тесты

```bash
conda activate tnnpm
cd /home/akursanov/models/tnnpm

# Все тесты
pytest

# Конкретный файл
pytest tests/test_model.py

# С покрытием
pytest --cov=src --cov-report=html
# Отчёт: htmlcov/index.html

# Конкретный класс тестов
pytest tests/test_model.py::TestRhsPhysics

# Быстрая проверка при разработке
pytest -x --tb=short
```

### Покрытие тестами

| Файл тестов | Что проверяется |
|---|---|
| `test_model.py` (67 тестов) | dataclass API, начальные условия, физические инварианты (`R+O+I+RI=1`, баланс натрия, сохранение заряда), ворота (зависимость `fCass` от `Ca_ss`), параллельная безопасность `rhs()` |
| `test_experiment.py` (42 теста) | `run_single`, цепочка прогонов, `run_sweep` (порядок результатов, декартово произведение), `sweep_combinations` |
| `test_io_plots.py` (45 тестов) | HDF5 roundtrip (time/variables/currents/forces/params), batch, legacy, все plot-функции |
| `test_cli.py` (27 тестов) | Парсинг `--set`, автоматическое определение sweep/single, `--warmup`, передача параметров |

---

## Архитектурные решения

| Решение | Причина |
|---|---|
| `rhs()` — чистая функция, без класса | Безопасен для параллельных sweep и оптимизаторов; нет скрытых побочных эффектов |
| `frozen=True` на всех dataclass | Случайная мутация параметров между прогонами невозможна |
| Токи и силы считаются в `calculate_outputs()` отдельно от `rhs()` | Солвер вызывает `rhs()` сотни тысяч раз — лишние вычисления накапливаются |
| `InitialState.from_result()` | Цепочка прогонов без ручного копирования 33 переменных |
| `scipy.optimize.brentq` вместо ручной бисекции | Гарантированная сходимость, контроль точности |
| Параметры CaMK вынесены в `TNNPMParams` | Все «магические числа» варьируемы; нет констант зашитых в `rhs()` |
| `simulation_io` вместо `io` | `io` — зарезервированное имя стандартной библиотеки Python |
| `conftest.py` мокирует Assimulo | Тесты физики и логики запускаются без установленного CVode |
| `n_out` в `SimConfig` отдельно от `time_stop` | Исправлена ошибка оригинала: `stop*1000` использовалось и как `tfinal`, и как число точек одновременно |
| `v_mech`, `w_mech` вместо `v`, `w` | Устранён конфликт имён с переменной `V` (потенциал); ошибки обнаруживаются статическим анализатором |
