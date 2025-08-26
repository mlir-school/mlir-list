#!/bin/bash

python3 -m venv mlir_venv
source mlir_venv/bin/activate
python -m pip install --upgrade pip

pip install numpy
pip install pybind11

