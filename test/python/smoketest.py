# RUN: %python %s pybind11 | FileCheck %s
# RUN: %python %s nanobind | FileCheck %s

import sys
from mlir_myproject.ir import *
from mlir_myproject.dialects import builtin as builtin_d

if sys.argv[1] == "pybind11":
    from mlir_myproject.dialects import mydialect_pybind11 as mydialect_d
elif sys.argv[1] == "nanobind":
    from mlir_myproject.dialects import mydialect_nanobind as mydialect_d
else:
    raise ValueError("Expected either pybind11 or nanobind as arguments")


with Context():
    mydialect_d.register_dialect()
    module = Module.parse(
        """
    %0 = arith.constant 2 : i32
    %1 = mydialect.foo %0 : i32
    """
    )
    # CHECK: %[[C:.*]] = arith.constant 2 : i32
    # CHECK: mydialect.foo %[[C]] : i32
    print(str(module))
