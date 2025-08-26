#!/bin/bash

source ./mlir_venv/bin/activate

mkdir -p install
INSTALL_DIR=$(realpath ./install)

mkdir build
cd build

# increase max number of open files for linking
ulimit -n 2048

#-DLLVM_EXTERNAL_LIT=$LLVM_BUILD_DIR/bin/llvm-lit \

cmake -G Ninja \
  -DMLIR_DIR=$LLVM_PREFIX/lib/cmake/mlir \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
  -DCMAKE_C_COMPILER=/usr/bin/clang \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
  -DCMAKE_EXE_LINKER_FLAGS_INIT="-fuse-ld=lld" \
  -DCMAKE_MODULE_LINKER_FLAGS_INIT="-fuse-ld=lld" \
  -DCMAKE_SHARED_LINKER_FLAGS_INIT="-fuse-ld=lld" \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DPython_FIND_VIRTUALENV=ONLY \
  -DPython3_FIND_VIRTUALENV=ONLY \
  -Dpybind11_DIR=`pybind11-config --cmakedir` \
  -DCMAKE_BUILD_TYPE=Debug \
  ..

cmake --build .
cmake --build . --target check-myproject
cmake --install .

