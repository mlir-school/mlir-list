# RUN: %python %s pybind11 | FileCheck %s
# RUN: %python %s nanobind | FileCheck %s

import sys
from mlir_listproject.ir import *
from mlir_listproject.dialects import builtin as builtin_d

if sys.argv[1] == "pybind11":
    from mlir_listproject.dialects import list_pybind11 as list_d
elif sys.argv[1] == "nanobind":
    from mlir_listproject.dialects import list_nanobind as list_d
else:
    raise ValueError("Expected either pybind11 or nanobind as arguments")


with Context():
    list_d.register_dialect()
    module = Module.parse(
        """
    %0 = arith.constant 2 : i32
    %1 = list.foo %0 : i32
    """
    )
    # CHECK: %[[C:.*]] = arith.constant 2 : i32
    # CHECK: list.foo %[[C]] : i32
    print(str(module))
