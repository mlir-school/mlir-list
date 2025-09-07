# RUN: %python %s | FileCheck %s

import sys
from mlir_listproject.ir import *
from mlir_listproject.dialects import builtin as builtin_d
from mlir_listproject.dialects import memref as memref
from mlir_listproject.dialects import arith as arith
import mlir_listproject.extras.types as T
from mlir_listproject.dialects.memref import _infer_memref_subview_result_type

from mlir_listproject.dialects import list_nanobind as list_d

with Context() as ctx, Location.unknown(ctx):
    module = Module.create()
    list_d.register_dialect()
    with InsertionPoint(module.body):
        c0 = arith.constant(T.i32(), 0)
        c100 = arith.constant(T.i32(), 100)
        # CHECK: %{{.*}} = list.range %{{.*}} to %{{.*}} : !list.list<i32>
        l0 = list_d.range(list_d.ListType.get(ctx, T.i32()), c0, c100)
        # CHECK: %{{.*}} = list.length %{{.*}} : !list.list<i32> -> i32
        length = list_d.length(l0)
        # CHECK: %{{.*}} = list.map %{{.*}} with (%{{.*}} : i32) -> i1 {
        mapOp = list_d.MapOp(list_d.ListType.get(ctx, T.i(1)), l0)
        mapOp.regions[0].blocks.append(T.i32())
        with InsertionPoint(mapOp.regions[0].blocks[0]):
            x = mapOp.regions[0].blocks[0].arguments[0]
            c42 = arith.constant(T.i32(), 42)
            pred = arith.cmpi(arith.CmpIPredicate.slt, x, c42)
            # CHECK: list.yield  %{{.*}} : i1
            list_d.yield_(pred)
    print(module)
