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
├── parameters.py      # TNNPMParams + InitialState + SimConfig (три frozen dataclass)
├── model.py           # TNNPM: rhs(), calculate_outputs(), make_init_state()
├── solver.py          # integrate() — тонкая обёртка CVode (Assimulo)
├── experiment.py      # SimulationResult + run_single(), run_sweep(), sweep_combinations()
├── simulation_io.py   # save/load HDF5 (single, batch, legacy, state)
├── plots.py           # визуализация (voltage, calcium, force, currents, overview, sweep)
├── configs/           # YAML-конфиги
│   ├── default.yaml
│   └── 2Hz_sweep_gCaL.yaml
└── scripts/
    └── run.py         # CLI точка входа (--config, --set, --warmup, --plot, --jobs)

tests/
├── conftest.py        # мок Assimulo на уровне sys.modules
├── test_model.py      # 67 тестов: dataclass, начальные условия, физика, gate CaL
├── test_experiment.py # 42 теста: run_single/run_sweep, chain-run, sweep_combinations
├── test_io_plots.py   # 58 тестов: HDF5 roundtrip, batch, legacy, все plot-функции
└── test_cli.py        # CLI тесты с моком solver/save
