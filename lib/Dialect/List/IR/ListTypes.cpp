//===- ListTypes.cpp - List dialect types -------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MyProject/Dialect/List/IR/ListTypes.h"

#include "MyProject/Dialect/List/IR/ListDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::list;

#define GET_TYPEDEF_CLASSES
#include "MyProject/Dialect/List/IR/ListOpsTypes.cpp.inc"

void ListDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "MyProject/Dialect/List/IR/ListOpsTypes.cpp.inc"
      >();
}
