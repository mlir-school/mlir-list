//===- MyDialectOps.cpp - MyDialect dialect ops -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MyProject/Dialect/MyDialect/IR/MyDialectOps.h"
#include "MyProject/Dialect/MyDialect/IR/MyDialectDialect.h"

#define GET_OP_CLASSES
#include "MyProject/Dialect/MyDialect/IR/MyDialectOps.cpp.inc"
