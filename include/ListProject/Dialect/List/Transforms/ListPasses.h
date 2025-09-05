//===- ListPasses.h - List passes  ------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LIST_LISTPASSES_H
#define LIST_LISTPASSES_H

#include "ListProject/Dialect/List/IR/ListDialect.h"
#include "ListProject/Dialect/List/IR/ListOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace list {
#define GEN_PASS_DECL
#include "ListProject/Dialect/List/Transforms/ListPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "ListProject/Dialect/List/Transforms/ListPasses.h.inc"
} // namespace list
} // namespace mlir

#endif
