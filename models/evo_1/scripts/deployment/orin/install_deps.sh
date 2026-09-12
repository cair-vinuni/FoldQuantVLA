#!/usr/bin/env bash
# Install this family's dependencies on a Jetson Orin (aarch64, JetPack 6.2,
# CUDA 12.6, Python 3.10). Modelled on
# models/groot_n1_7/scripts/deployment/orin/install_deps.sh.
#
# No sudo and no apt: everything lands in the virtualenv. TensorRT is the one
# thing that cannot come from an index — JetPack ships it as a system package —
# so it is made importable with a .pth file rather than installed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ]; then
    echo "ERROR: this script is for aarch64 (Jetson Orin). Detected: $ARCH"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [ "$PYTHON_VERSION" != "3.10" ]; then
    echo "ERROR: JetPack 6.2 ships Python 3.10 and the jetson-jp6-cu126 index"
    echo "       publishes cp310 wheels only. Detected: $PYTHON_VERSION"
    exit 1
fi

if ! command -v uv &> /dev/null; then
    echo "ERROR: uv not found. See https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

# Resolve the platform pyproject into a venv without installing the project
# itself: the integration is loaded with PYTHONPATH, so an editable install
# would only write .egg-info into the source tree.
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$REPO_ROOT/.venv}"
echo "uv sync from $SCRIPT_DIR into $UV_PROJECT_ENVIRONMENT"
uv sync --project "$SCRIPT_DIR" --no-install-project

VENV_PYTHON="$UV_PROJECT_ENVIRONMENT/bin/python"
SITE_PKGS="$UV_PROJECT_ENVIRONMENT/lib/python${PYTHON_VERSION}/site-packages"

# torch 2.10.0 wants libcudss.so.0 at runtime. --no-deps keeps it from pulling
# nvidia-cublas-cu12, which collides with JetPack's own CUDA 12.6 libraries.
echo "Installing nvidia-cudss-cu12 (no-deps)..."
uv pip install --python "$VENV_PYTHON" --no-deps nvidia-cudss-cu12

# JetPack's TensorRT lives in the system dist-packages and is on no index.
echo "Linking JetPack system packages (TensorRT) into the venv..."
echo "/usr/lib/python${PYTHON_VERSION}/dist-packages" \
    > "$SITE_PKGS/jetpack-system-packages.pth"

echo
echo "Done. Use it with:"
echo "  export PYTHONPATH=\$(cd "$REPO_ROOT/../.." && pwd):$REPO_ROOT"
echo "  $VENV_PYTHON -m foldquant_integration.export_foldquant --help"
echo
echo "Plugin libraries are built separately and cached per (SM, machine, TensorRT):"
echo "  $VENV_PYTHON -m foldquant.kernels build && $VENV_PYTHON -m foldquant.kernels status"
