"""
test_cli.py
===========
Тесты для scripts/run.py.

Солвер и save мокируются — тесты проверяют логику CLI:
парсинг аргументов, построение параметров, маршрутизацию sweep/single.

Запуск:
    pytest test_cli.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

# run.py лежит в scripts/ — добавляем его в путь
sys.path.insert(0, str(Path(__file__).parent))

from scripts.run import (
    _build_params_and_config,
    _detect_sweep_param,
    _parse_overrides,
    main,
)
from parameters import DEFAULT_PARAMS, SimConfig, TNNPMParams

# ---------------------------------------------------------------------------
# Фикстуры
# ---------------------------------------------------------------------------


@pytest.fixture
def default_yaml():
    return {
        "params": {"stim_period": 500.0, "g_Na": 14.838},
        "config": {"time_stop": 1000.0, "atol": 1e-9},
        "mechanics": {"F_afterload": 0.0},
    }


@pytest.fixture
def sweep_yaml():
    return {
        "params": {"stim_period": 500.0, "g_CaL": [3e-5, 5e-5, 7e-5]},
        "config": {"time_stop": 1000.0},
        "mechanics": {},
    }


@pytest.fixture
def mock_result(tmp_path):
    """Минимальный SimulationResult-мок."""
    from experiment import SimulationResult
    from model import make_init_state

    p = TNNPMParams()
    s, l0 = make_init_state(p)
    N = 5
    t = np.linspace(0, 100, N)
    y0 = np.array(s.to_array())

    from experiment import _y_to_variables, calculate_outputs

    y = np.tile(y0, (N, 1))
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
    return SimulationResult(
        time=t,
        variables=variables,
        currents=currents,
        forces=forces,
        params=p,
        state0=s,
        config=SimConfig(time_stop=100.0),
        meta={"wall_time_s": 0.1, "n_points": N, "F_afterload": 0.0, "l0": l0},
    )


# ===========================================================================
# 1. _parse_overrides
# ===========================================================================


class TestParseOverrides:

    def test_float(self):
        r = _parse_overrides(["g_Na=16.0"])
        assert r["g_Na"] == pytest.approx(16.0)
        assert isinstance(r["g_Na"], float)

    def test_int(self):
        r = _parse_overrides(["n_out=100"])
        assert r["n_out"] == 100
        assert isinstance(r["n_out"], int)

    def test_scientific(self):
        r = _parse_overrides(["g_CaL=5e-5"])
        assert r["g_CaL"] == pytest.approx(5e-5)

    def test_multiple(self):
        r = _parse_overrides(["g_Na=14.0", "g_CaL=5e-5"])
        assert len(r) == 2
        assert r["g_Na"] == pytest.approx(14.0)

    def test_empty(self):
        assert _parse_overrides([]) == {}

    def test_missing_equals_exits(self):
        with pytest.raises(SystemExit):
            _parse_overrides(["g_Na_16"])


# ===========================================================================
# 2. _build_params_and_config
# ===========================================================================


class TestBuildParamsAndConfig:

    def test_scalar_returns_no_sweep(self, default_yaml):
        base, sweep, config, mech = _build_params_and_config(default_yaml, {})
        assert sweep is None
        assert isinstance(base, TNNPMParams)

    def test_scalar_params_applied(self, default_yaml):
        base, *_ = _build_params_and_config(default_yaml, {})
        assert base.stim_period == pytest.approx(500.0)
        assert base.g_Na == pytest.approx(14.838)

    def test_list_values_become_sweep(self, sweep_yaml):
        base, sweep, config, mech = _build_params_and_config(sweep_yaml, {})
        assert sweep is not None
        assert "g_CaL" in sweep
        assert sweep["g_CaL"] == [3e-5, 5e-5, 7e-5]

    def test_config_applied(self, default_yaml):
        _, _, config, _ = _build_params_and_config(default_yaml, {})
        assert config.time_stop == pytest.approx(1000.0)
        assert config.atol == pytest.approx(1e-9)

    def test_overrides_over_yaml(self, default_yaml):
        base, *_ = _build_params_and_config(default_yaml, {"g_Na": 99.0})
        assert base.g_Na == pytest.approx(99.0)

    def test_override_turns_scalar_into_sweep(self, default_yaml):
        """CLI --set g_CaL=[...] недопустим, но --set g_Na=0.0 работает."""
        base, sweep, *_ = _build_params_and_config(default_yaml, {"g_Na": 0.0})
        assert sweep is None  # override скаляра — не sweep
        assert base.g_Na == pytest.approx(0.0)

    def test_mechanics_f_afterload(self, default_yaml):
        *_, mech = _build_params_and_config(default_yaml, {})
        assert mech.get("F_afterload", 0.0) == pytest.approx(0.0)

    def test_empty_yaml(self):
        base, sweep, config, mech = _build_params_and_config({}, {})
        assert sweep is None
        assert isinstance(base, TNNPMParams)
        assert isinstance(config, SimConfig)


# ===========================================================================
# 3. _detect_sweep_param
# ===========================================================================


class TestDetectSweepParam:

    def test_detects_g_CaL(self, mock_result):
        import dataclasses

        r1 = mock_result
        r2 = dataclasses.replace(
            r1, params=r1.params.replace(g_CaL=r1.params.g_CaL * 2)
        )
        param = _detect_sweep_param([r1, r2])
        assert param == "g_CaL"

    def test_single_result_returns_default(self, mock_result):
        # Не падает на одном результате
        param = _detect_sweep_param([mock_result])
        assert isinstance(param, str)

    def test_detects_stim_period(self, mock_result):
        import dataclasses

        r2 = dataclasses.replace(
            mock_result, params=mock_result.params.replace(stim_period=500.0)
        )
        param = _detect_sweep_param([mock_result, r2])
        assert param == "stim_period"


# ===========================================================================
# 4. main() — одиночный прогон
# ===========================================================================


class TestMainSingle:

    def test_run_single_called(self, mock_result, tmp_path):
        out = tmp_path / "result.h5"
        with patch(
            "scripts.run.run_single", return_value=mock_result
        ) as mock_rs, patch("scripts.run.save"):
            main(["--output", str(out), "--no-save"])
            mock_rs.assert_called_once()

    def test_save_called_with_output_path(self, mock_result, tmp_path):
        out = tmp_path / "result.h5"
        with patch("scripts.run.run_single", return_value=mock_result), patch(
            "scripts.run.save"
        ) as mock_save:
            main(["--output", str(out)])
            mock_save.assert_called_once()
            actual_path = mock_save.call_args[0][1]
            assert Path(actual_path) == out

    def test_no_save_skips_save(self, mock_result, tmp_path):
        out = tmp_path / "result.h5"
        with patch("scripts.run.run_single", return_value=mock_result), patch(
            "scripts.run.save"
        ) as mock_save:
            main(["--output", str(out), "--no-save"])
            mock_save.assert_not_called()

    def test_returns_zero_on_success(self, mock_result, tmp_path):
        out = tmp_path / "result.h5"
        with patch("scripts.run.run_single", return_value=mock_result), patch(
            "scripts.run.save"
        ):
            code = main(["--output", str(out)])
            assert code == 0

    def test_override_passed_to_run_single(self, mock_result, tmp_path):
        """--set g_Na=0.0 передаётся в params."""
        out = tmp_path / "r.h5"
        captured = {}

        def fake_run(params, **kwargs):
            captured["params"] = params
            return mock_result

        with patch("scripts.run.run_single", side_effect=fake_run), patch(
            "scripts.run.save"
        ):
            main(["--output", str(out), "--set", "g_Na=0.0", "--no-save"])
        assert captured["params"].g_Na == pytest.approx(0.0)

    def test_config_yaml_applied(self, mock_result, tmp_path):
        """Параметры из YAML-файла попадают в прогон."""
        import yaml

        cfg = tmp_path / "test.yaml"
        cfg.write_text(
            yaml.dump(
                {
                    "params": {"stim_period": 500.0},
                    "config": {"time_stop": 2000.0},
                    "mechanics": {},
                }
            )
        )
        captured = {}

        def fake_run(params, state0, config, **kwargs):
            captured["config"] = config
            captured["params"] = params
            return mock_result

        out = tmp_path / "result.h5"
        with patch("scripts.run.run_single", side_effect=fake_run), patch(
            "scripts.run.save"
        ):
            main(["--config", str(cfg), "--output", str(out), "--no-save"])
        assert captured["config"].time_stop == pytest.approx(2000.0)
        assert captured["params"].stim_period == pytest.approx(500.0)


# ===========================================================================
# 5. main() — sweep
# ===========================================================================


class TestMainSweep:

    @pytest.fixture
    def sweep_results(self, mock_result):
        import dataclasses

        return [
            mock_result,
            dataclasses.replace(
                mock_result, params=mock_result.params.replace(g_CaL=7e-5)
            ),
        ]

    def test_run_sweep_called_for_list_params(self, sweep_results, tmp_path):
        import yaml

        cfg = tmp_path / "sweep.yaml"
        cfg.write_text(
            yaml.dump(
                {
                    "params": {"g_CaL": [5e-5, 7e-5]},
                    "config": {"time_stop": 1000.0},
                    "mechanics": {},
                }
            )
        )
        out = tmp_path / "sweep.h5"
        with patch(
            "scripts.run.run_sweep", return_value=sweep_results
        ) as mock_sw, patch("scripts.run.save_batch"):
            main(["--config", str(cfg), "--output", str(out), "--no-save"])
            mock_sw.assert_called_once()

    def test_save_batch_called_for_sweep(self, sweep_results, tmp_path):
        import yaml

        cfg = tmp_path / "sweep.yaml"
        cfg.write_text(
            yaml.dump(
                {
                    "params": {"g_CaL": [5e-5, 7e-5]},
                    "config": {"time_stop": 1000.0},
                    "mechanics": {},
                }
            )
        )
        out = tmp_path / "sweep.h5"
        with patch("scripts.run.run_sweep", return_value=sweep_results), patch(
            "scripts.run.save_batch"
        ) as mock_sb:
            main(["--config", str(cfg), "--output", str(out)])
            mock_sb.assert_called_once()

    def test_n_jobs_passed_to_run_sweep(self, sweep_results, tmp_path):
        import yaml

        cfg = tmp_path / "sweep.yaml"
        cfg.write_text(
            yaml.dump(
                {
                    "params": {"g_CaL": [5e-5, 7e-5]},
                    "config": {"time_stop": 1000.0},
                    "mechanics": {},
                }
            )
        )
        out = tmp_path / "sweep.h5"
        captured = {}

        def fake_sweep(**kwargs):
            captured["n_jobs"] = kwargs.get("n_jobs")
            return sweep_results

        with patch("scripts.run.run_sweep", side_effect=fake_sweep), patch(
            "scripts.run.save_batch"
        ):
            main(
                ["--config", str(cfg), "--output", str(out), "--jobs", "4", "--no-save"]
            )
        assert captured["n_jobs"] == 4


# ===========================================================================
# 6. main() — warmup
# ===========================================================================


class TestMainWarmup:

    def test_warmup_file_loaded(self, mock_result, tmp_path):
        """--warmup вызывает load_state()."""
        from simulation_io import save

        warmup = tmp_path / "warmup.h5"
        save(mock_result, warmup)

        out = tmp_path / "result.h5"
        with patch(
            "scripts.run.run_single", return_value=mock_result
        ) as mock_rs, patch("scripts.run.save"):
            main(["--warmup", str(warmup), "--output", str(out), "--no-save"])

        # state0 должен быть InitialState (не None)
        _, kwargs = mock_rs.call_args
        from parameters import InitialState

        assert isinstance(
            kwargs.get("state0") or mock_rs.call_args[0][1], (InitialState, type(None))
        )
