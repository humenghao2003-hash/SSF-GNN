#!/bin/bash
# Launcher for the SSF-GNN training environment.
# - Uses the project venv (torch 2.9.0+cu128)
# - Exposes GDAL's shared libraries (conda env) to the dynamic linker
set -e
cd "$(dirname "$0")"
export LD_LIBRARY_PATH="/home/ubuntu/miniconda3/envs/gdal-env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$(pwd)/.venv/bin/python" "$@"
