//===- MyDialectPasses.h - MyDialect passes  ------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MYDIALECT_MYDIALECTPASSES_H
#define MYDIALECT_MYDIALECTPASSES_H

#include "MyProject/Dialect/MyDialect/IR/MyDialectDialect.h"
#include "MyProject/Dialect/MyDialect/IR/MyDialectOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace mydialect {
#define GEN_PASS_DECL
#include "MyProject/Dialect/MyDialect/Transforms/MyDialectPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "MyProject/Dialect/MyDialect/Transforms/MyDialectPasses.h.inc"
} // namespace mydialect
} // namespace mlir

#endif
