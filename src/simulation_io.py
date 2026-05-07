"""
simulation_io.py
=====
Сохранение и загрузка результатов симуляции.

Публичный API:
    save(result, path)           — сохранить SimulationResult в HDF5
    load(path)  -> SimulationResult  — загрузить из HDF5
    load_state(path) -> InitialState — только последняя точка (для цепочки прогонов)

Формат файла HDF5:
    /time                        — массив времён (N,)
    /variables/<name>            — фазовые переменные
    /currents/<name>             — ионные токи
    /forces/<name>               — механические силы
    /params/<name>               — скаляры TNNPMParams
    /state0/<name>               — скаляры InitialState
    /config/<name>               — скаляры SimConfig
    /meta/<name>                 — служебные данные

Совместимость с оригинальным форматом:
    load_legacy(path) -> dict    — читает файлы, сохранённые старым saveh5()
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import h5py

from experiment import SimulationResult
from parameters import InitialState, SimConfig, TNNPMParams

# ---------------------------------------------------------------------------
# Сохранение
# ---------------------------------------------------------------------------


def save(result: SimulationResult, path: str | Path) -> None:
    """Сохранить SimulationResult в HDF5.

    Пример:
        save(result, 'data/2Hz_1000c.h5')
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        # Время
        f.create_dataset("time", data=result.time)

        # Фазовые переменные
        grp = f.create_group("variables")
        for name, arr in result.variables.items():
            grp.create_dataset(name, data=arr)

        # Токи
        grp = f.create_group("currents")
        for name, arr in result.currents.items():
            grp.create_dataset(name, data=arr)

        # Силы
        grp = f.create_group("forces")
        for name, arr in result.forces.items():
            grp.create_dataset(name, data=arr)

        # Параметры модели
        _write_dataclass(f.create_group("params"), result.params)

        # Начальные условия
        _write_dataclass(f.create_group("state0"), result.state0)

        # Конфигурация прогона
        _write_dataclass(f.create_group("config"), result.config)

        # Мета (только скаляры/строки)
        grp = f.create_group("meta")
        for k, v in result.meta.items():
            try:
                grp.attrs[k] = v
            except TypeError:
                grp.attrs[k] = str(v)


def save_batch(results: list[SimulationResult], path: str | Path) -> None:
    """Сохранить список результатов sweep в один HDF5.

    Структура: /run_0/, /run_1/, ... — каждый содержит те же группы что save().

    Пример:
        results = run_sweep(sweep={'g_CaL': [3e-5, 5e-5, 7e-5]}, ...)
        save_batch(results, 'data/sweep_gCaL.h5')
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        f.attrs["n_runs"] = len(results)
        for i, result in enumerate(results):
            grp = f.create_group(f"run_{i}")
            _write_result_to_group(grp, result)


# ---------------------------------------------------------------------------
# Загрузка
# ---------------------------------------------------------------------------


def load(path: str | Path) -> SimulationResult:
    """Загрузить SimulationResult из HDF5.

    Пример:
        result = load('data/2Hz_1000c.h5')
        plt.plot(result.time, result.V)
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        return _read_result_from_group(f)


def load_batch(path: str | Path) -> list[SimulationResult]:
    """Загрузить все прогоны из файла save_batch().

    Пример:
        results = load_batch('data/sweep_gCaL.h5')
        for r in results:
            print(r.params.g_CaL, r.Ca_i.max())
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        n = f.attrs.get("n_runs", 0)
        return [_read_result_from_group(f[f"run_{i}"]) for i in range(n)]


def load_state(path: str | Path) -> InitialState:
    """Загрузить только последнюю точку прогона как InitialState.

    Удобно для цепочки прогонов без загрузки всего файла в память:
        state0 = load_state('data/warmup.h5')
        result = run_single(params, state0, config_main)
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        variables = {k: f["variables"][k][...] for k in f["variables"]}
    return InitialState.from_result(variables)


def load_legacy(path: str | Path) -> dict:
    """Загрузить файл в старом формате (saveh5 из utils.py).

    Возвращает сырой словарь — совместимость с оригинальным кодом.
    Используйте load() для новых файлов.

    Пример:
        d = load_legacy('data/resultsST.h5')
        state0 = InitialState.from_result(d['variables'])
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        return _read_group_recursive(f)


# ---------------------------------------------------------------------------
# Загрузка YAML-конфига
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> tuple[TNNPMParams, SimConfig]:
    """Загрузить параметры и конфигурацию из YAML-файла.

    Формат YAML:
        params:
          g_CaL: 5.0e-5
          stim_period: 500.0
        config:
          time_stop: 500000.0
          atol: 1.0e-9

    Пример:
        params, config = load_config('configs/2Hz.yaml')
        result = run_single(params, state0, config)
    """
    try:
        import yaml
    except ImportError as e:
        raise ImportError("PyYAML не установлен. Установите: pip install pyyaml") from e

    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f)

    params = TNNPMParams.from_dict(data.get("params", {}))
    config = SimConfig.from_dict(data.get("config", {}))
    return params, config


# ---------------------------------------------------------------------------
# Приватные утилиты
# ---------------------------------------------------------------------------


def _write_dataclass(grp: h5py.Group, obj) -> None:
    """Записать поля dataclass как атрибуты группы."""
    for fld in dataclasses.fields(obj):
        val = getattr(obj, fld.name)
        if val is None:
            grp.attrs[fld.name] = "__None__"
        else:
            try:
                grp.attrs[fld.name] = val
            except TypeError:
                grp.attrs[fld.name] = str(val)


def _read_dataclass(grp: h5py.Group, cls):
    """Восстановить dataclass из атрибутов группы."""
    known = {f.name: f for f in dataclasses.fields(cls)}
    kwargs = {}
    for k, v in grp.attrs.items():
        if k not in known:
            continue
        if v == "__None__":
            kwargs[k] = None
        else:
            # Определяем базовый тип из аннотации, поддерживая union (int | None)
            ann = known[k].type
            ann_str = str(ann)
            # Извлекаем первый тип из union: "int | None" → "int"
            base_type = ann_str.split("|")[0].strip()
            try:
                if base_type in ("int", "<class 'int'>") or ann is int:
                    kwargs[k] = int(v)
                elif base_type in ("float", "<class 'float'>") or ann is float:
                    kwargs[k] = float(v)
                else:
                    kwargs[k] = v
            except (TypeError, ValueError):
                kwargs[k] = v
    return cls.from_dict(kwargs)


def _write_result_to_group(grp: h5py.Group, result: SimulationResult) -> None:
    grp.create_dataset("time", data=result.time)
    for sub, data in [
        ("variables", result.variables),
        ("currents", result.currents),
        ("forces", result.forces),
    ]:
        sg = grp.create_group(sub)
        for name, arr in data.items():
            sg.create_dataset(name, data=arr)
    _write_dataclass(grp.create_group("params"), result.params)
    _write_dataclass(grp.create_group("state0"), result.state0)
    _write_dataclass(grp.create_group("config"), result.config)
    meta_grp = grp.create_group("meta")
    for k, v in result.meta.items():
        try:
            meta_grp.attrs[k] = v
        except TypeError:
            meta_grp.attrs[k] = str(v)


def _read_result_from_group(grp: h5py.Group) -> SimulationResult:
    time = grp["time"][...]
    variables = {k: grp["variables"][k][...] for k in grp["variables"]}
    currents = {k: grp["currents"][k][...] for k in grp["currents"]}
    forces = {k: grp["forces"][k][...] for k in grp["forces"]}
    params = _read_dataclass(grp["params"], TNNPMParams)
    state0 = _read_dataclass(grp["state0"], InitialState)
    config = _read_dataclass(grp["config"], SimConfig)
    meta = dict(grp["meta"].attrs)
    return SimulationResult(
        time=time,
        variables=variables,
        currents=currents,
        forces=forces,
        params=params,
        state0=state0,
        config=config,
        meta=meta,
    )


def _read_group_recursive(grp) -> dict:
    """Рекурсивно читать HDF5 группу в словарь (совместимость со старым форматом)."""
    result = {}
    for key in grp.keys():
        if isinstance(grp[key], h5py.Group):
            result[key] = _read_group_recursive(grp[key])
        else:
            result[key] = grp[key][...]
    return result
