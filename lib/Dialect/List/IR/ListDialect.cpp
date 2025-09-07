//===- ListDialect.cpp - List dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ListProject/Dialect/List/IR/ListDialect.h"
#include "ListProject/Dialect/List/IR/ListOps.h"

using namespace mlir;
using namespace mlir::list;

#include "ListProject/Dialect/List/IR/ListOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// List dialect.
//===----------------------------------------------------------------------===//

void ListDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ListProject/Dialect/List/IR/ListOps.cpp.inc"
      >();
}
