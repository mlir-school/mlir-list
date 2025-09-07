# RUN: %python %s | FileCheck %s

import sys
from mlir_listproject.ir import *
from mlir_listproject.dialects import builtin as builtin_d

from mlir_listproject.dialects import list_nanobind as list_d

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
