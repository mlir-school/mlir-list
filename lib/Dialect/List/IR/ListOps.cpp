//===- ListOps.cpp - List dialect ops -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Block.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/ValueRange.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

#include "ListProject/Dialect/List/IR/ListOps.h"
#include "ListProject/Dialect/List/IR/ListDialect.h"
#include "ListProject/Dialect/List/IR/ListTypes.h"

#define GET_OP_CLASSES
#include "ListProject/Dialect/List/IR/ListOps.cpp.inc"

using namespace mlir;
using namespace list;
