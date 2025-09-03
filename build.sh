#!/bin/bash

mkdir -p install
INSTALL_DIR=$(realpath ./install)

mkdir -p build
cd build

# increase max number of open files for linking
ulimit -n 2048

LLVM_PREFIX=$(python -c 'import mlir;print(mlir.__path__[0])')

cmake -G Ninja \
  -DMLIR_DIR=$LLVM_PREFIX/lib/cmake/mlir \
  -DLLVM_EXTERNAL_LIT=$(which lit) \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
  -DCMAKE_C_COMPILER=/usr/bin/clang \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DPython_FIND_VIRTUALENV=ONLY \
  -DPython3_FIND_VIRTUALENV=ONLY \
  -Dpybind11_DIR=`pybind11-config --cmakedir` \
  -DLLVM_ENABLE_LLD=On \
  -DLLVM_BUILD_LLVM_DYLIB=ON \
  -DLLVM_LINK_LLVM_DYLIB=ON \
  -DCMAKE_BUILD_TYPE=Debug \
  ..

cmake --build . --target check-myproject

