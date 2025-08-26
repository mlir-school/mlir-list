#!/bin/bash

source ./mlir_venv/bin/activate

cd llvm-project

ulimit -n 2048

python -m pip install -r mlir/python/requirements.txt

mkdir -p ./install
cmake -GNinja -Bbuild \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DLLVM_BUILD_EXAMPLES=False \
  -DCMAKE_INSTALL_PREFIX=./install \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DPython3_FIND_VIRTUALENV=ONLY \
  -DPython_FIND_VIRTUALENV=ONLY \
  -DPython3_ROOT_DIR=../venv/mlir_venv \
  -DLLVM_TARGETS_TO_BUILD="Native" \
  -DLLVM_INSTALL_UTILS=ON \
  -DLLVM_PARALLEL_LINK_JOBS=4 \
  -DLLVM_PARALLEL_COMPILE_JOBS="$(nproc)" \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_C_COMPILER=clang \
  -DLLVM_USE_SPLIT_DWARF=On \
  -DLLVM_ENABLE_ASSERTIONS=On \
  -DLLVM_ENABLE_LLD=On \
  -DLLVM_BUILD_LLVM_DYLIB=ON \
  -DLLVM_LINK_LLVM_DYLIB=ON \
  -DMLIR_LINK_MLIR_DYLIB=ON \
  -DMLIR_BUILD_MLIR_C_DYLIB=ON \
  ./llvm

cmake --build build --target mlir-opt mlir-translate mlir-transform-opt mlir-cpu-runner check-mlir
cmake --build build --target install
cd ..

