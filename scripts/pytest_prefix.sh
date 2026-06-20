#!/usr/bin/env bash
set -euo pipefail

unset PYTHONPATH
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

exec uv run python -m pytest "$@"