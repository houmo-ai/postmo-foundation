#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
WHEEL_PATH="${1:-}"
INSTALL_MODE="${INSTALL_MODE:-no-deps}"
INSTALL_DIFFUSERS="${INSTALL_DIFFUSERS:-0}"
TCIM_LITE_SOURCE="${TCIM_LITE_SOURCE:-}"
VLLM_HOUMO_SOURCE="${VLLM_HOUMO_SOURCE:-}"

if [[ -z "$WHEEL_PATH" ]]; then
  WHEEL_PATH="$(find dist -maxdepth 1 -name 'hliepython-*.whl' | sort | tail -n 1)"
fi

if [[ -z "$WHEEL_PATH" || ! -f "$WHEEL_PATH" ]]; then
  echo "ERROR: wheel not found. Build one first with: python -m build --wheel" >&2
  exit 1
fi

echo "Using python: $PYTHON_BIN"
echo "Using wheel : $WHEEL_PATH"

"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install inflect

if [[ -n "$TCIM_LITE_SOURCE" ]]; then
  echo "Installing tcim_lite from: $TCIM_LITE_SOURCE"
  "$PYTHON_BIN" -m pip install "$TCIM_LITE_SOURCE"
fi

if [[ -n "$VLLM_HOUMO_SOURCE" ]]; then
  echo "Installing vllm_houmo from: $VLLM_HOUMO_SOURCE"
  "$PYTHON_BIN" -m pip install "$VLLM_HOUMO_SOURCE"
fi

if [[ "$INSTALL_DIFFUSERS" == "1" ]]; then
  "$PYTHON_BIN" -m pip install diffusers
fi

if [[ "$INSTALL_MODE" == "with-deps" ]]; then
  "$PYTHON_BIN" -m pip install --force-reinstall "$WHEEL_PATH"
else
  "$PYTHON_BIN" -m pip install --force-reinstall --no-deps "$WHEEL_PATH"
fi

VERIFY_DIR="$(mktemp -d)"

(
cd "$VERIFY_DIR"
"$PYTHON_BIN" - <<'PY'
import importlib
import importlib.resources as ir

modules = [
    "hliepython",
    "hliepython.main",
    "hliepython.model_server.inference_adapter",
]
for name in modules:
    mod = importlib.import_module(name)
    print(f"import ok: {name} -> {getattr(mod, '__file__', 'built-in')}")

resources = [
    ir.files("hliepython").joinpath("config.yaml"),
    ir.files("hliepython.model_server.engines").joinpath("zero_shot_prompt.wav"),
    ir.files("hliepython.model_server.engines.files").joinpath("special_tokens.json"),
]
for path in resources:
    print(f"resource ok: {path} -> {path.is_file()}")
PY
)

rm -rf "$VERIFY_DIR"

echo "Install completed."