# VENV

if [ -d "./mlir_venv" ]; then
  source ./mlir_venv/bin/activate
fi

# LLVM-PROJECT

export PATH=$(realpath ./llvm-project/build/bin):$PATH

# llvm-project build dir
export LLVM_BUILD_DIR="$(realpath ./llvm-project/build)"
# llvm-project install dir
export LLVM_PREFIX="$(realpath ./llvm-project/install)"
# mlir python bindings
export PYTHONPATH=$(realpath ./llvm-project/install/python_packages/mlir_core):$PYTHONPATH

# MYPROJECT

# myproject binaries
export PATH=$(realpath ./build/bin):$PATH

# get built python
export PYTHONPATH=$(realpath ./build/python_packages/myproject):$PYTHONPATH

