"""conftest.py — мокирует assimulo до импорта тест-модулей."""

import sys
from unittest.mock import MagicMock

for mod in ["assimulo", "assimulo.problem", "assimulo.solvers"]:
    sys.modules[mod] = MagicMock()
