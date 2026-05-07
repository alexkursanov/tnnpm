# tnnpm

Описание проекта.

## Установка

```bash
npm install
```

## Использование

```bash
npm start
```

src/
├── parameters.py      # TNNPMParams + InitialState (два dataclass)
├── model.py           # TNNPM: rhs(), calculate_outputs()
├── solver.py          # integrate() — чистая функция
├── postprocessing.py  # SimulationResult + вычисление токов/сил
├── experiment.py      # run_single(), run_sweep(), run_batch()
├── io.py              # save/load HDF5, load YAML
└── plots.py           # функции над SimulationResult

scripts/
├── run.py             # CLI точка входа
└── sweep_example.py   # пример параметрического прогона

notebooks/
└── analysis.ipynb

tests/
├── test_model.py      # тесты на физику: сохранение заряда и т.д.
└── test_io.py

configs/
├── default.yaml
└── 2Hz_sweep.yaml
