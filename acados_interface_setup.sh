#!/usr/bin/env bash

# Source this file to enable local acados runtime for concrete_block_motion_planning.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ACADOS_SOURCE_DIR="$SCRIPT_DIR/acados"

if [ -d "$ACADOS_SOURCE_DIR/lib" ]; then
  export LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib:${LD_LIBRARY_PATH:-}"
fi

if [ -d "$ACADOS_SOURCE_DIR/interfaces/acados_template" ]; then
  export PYTHONPATH="$ACADOS_SOURCE_DIR/interfaces/acados_template:${PYTHONPATH:-}"
fi

if [ -d "$ACADOS_SOURCE_DIR/bin" ]; then
  export PATH="$ACADOS_SOURCE_DIR/bin:${PATH}"
fi

echo "ACADOS_SOURCE_DIR=$ACADOS_SOURCE_DIR"
if command -v t_renderer >/dev/null 2>&1; then
  echo "t_renderer=$(command -v t_renderer)"
else
  echo "t_renderer not found in PATH"
fi
