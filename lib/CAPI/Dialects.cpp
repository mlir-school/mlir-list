//===- Dialects.cpp - CAPI for dialects -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ListProject-c/Dialects.h"

#include "ListProject/Dialect/List/IR/ListDialect.h"
#include "ListProject/Dialect/List/IR/ListTypes.h"
#include "mlir/CAPI/Registration.h"

using namespace mlir;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(List, list,
                                      mlir::list::ListDialect)

//===-------------------------------------------------------------------===//
// ListType
//===-------------------------------------------------------------------===//

bool mlirTypeIsAListType(MlirType type) {
  return isa<list::ListType>(unwrap(type));
}

MlirType mlirListTypeGet(MlirContext ctx, MlirType elementType) {
  return wrap(list::ListType::get(unwrap(ctx), unwrap(elementType)));
}

