//===- ListPasses.cpp - List passes -------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "ListProject/Dialect/List/Transforms/ListPasses.h"

namespace mlir::list {
#define GEN_PASS_DEF_LISTREMOVESOMEFOO
#include "ListProject/Dialect/List/Transforms/ListPasses.h.inc"

namespace {
class ListRemoveSomeFoo
    : public impl::ListRemoveSomeFooBase<ListRemoveSomeFoo> {
public:
  using impl::ListRemoveSomeFooBase<
      ListRemoveSomeFoo>::ListRemoveSomeFooBase;
  void runOnOperation() final {
    // Your pass code here
    // ======================================================
    ModuleOp module = getOperation();
    module->walk([&](list::FooOp fooOp) {
      if (fooOp->hasAttr("useless")) {
	fooOp.replaceAllUsesWith(fooOp.getInput());
	fooOp.erase();
      }
    });
    // ======================================================
  }
};
} // namespace
} // namespace mlir::list
