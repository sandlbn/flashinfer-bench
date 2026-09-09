#!/bin/bash
set -eo pipefail
set -x
echo "Linting..."

# ruff must already be present: this script does not install anything on its own.
# Installing is the environment owner's decision (see CLAUDE.md, "Python Environments").
if ! command -v ruff &> /dev/null; then
    echo "ruff not found on PATH. Install it into the active environment yourself" >&2
    echo "(e.g. 'python -m pip install --no-deps ruff'), then re-run." >&2
    exit 1
fi

ruff check . --fix
