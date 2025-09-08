//===- Dialects.h - CAPI for dialects -----------------------------*- C -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MYPROJECT_C_DIALECTS_H
#define MYPROJECT_C_DIALECTS_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(List, list);

//===-------------------------------------------------------------------===//
// ListType
//===-------------------------------------------------------------------===//

REMOVE_ME!!! Declare CAPI wrappers for !list.list<> type !!!REMOVE_ME
MLIR_CAPI_EXPORTED bool mlirTypeIsAListType(MlirType type);

MLIR_CAPI_EXPORTED MlirType mlirListTypeGet(MlirContext ctx, MlirType elementType);

#ifdef __cplusplus
}
#endif

#endif // MYPROJECT_C_DIALECTS_H
