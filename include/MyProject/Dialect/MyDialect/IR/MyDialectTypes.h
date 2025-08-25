//===- MyDialectTypes.h - MyDialect dialect types -------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MYDIALECT_MYDIALECTTYPES_H
#define MYDIALECT_MYDIALECTTYPES_H

#include "mlir/IR/BuiltinTypes.h"

#define GET_TYPEDEF_CLASSES
#include "MyProject/Dialect/MyDialect/IR/MyDialectOpsTypes.h.inc"

#endif // MYDIALECT_MYDIALECTTYPES_H
