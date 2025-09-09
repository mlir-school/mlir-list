//===- ListProjectExtension.cpp - Extension module --------------------------===//
//
// This is the nanobind version of the example module. It's also possible to use
// PyBind11
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ListProject-c/Dialects.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

namespace nb = nanobind;

using namespace mlir;
using namespace mlir::python;
using namespace mlir::python::nanobind_adaptors;

NB_MODULE(_listprojectDialectsNanobind, m) {
  auto listM = m.def_submodule("list");

  //===-------------------------------------------------------------------===//
  // ListType
  //===-------------------------------------------------------------------===//

  auto mlirListType =
      mlir_type_subclass(m, "ListType", mlirTypeIsAListType);

  mlirListType.def_classmethod(
      "get",
      [](const nb::object &cls, MlirContext ctx, MlirType element_type) {
        return cls(mlirListTypeGet(ctx, element_type));
      },
      "Gets an instance of ListType in the same context", nb::arg("cls"),
      nb::arg("ctx"),
      nb::arg("element_type"));

  //===--------------------------------------------------------------------===//
  // list dialect
  //===--------------------------------------------------------------------===//
  //
  listM.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle handle = mlirGetDialectHandle__list__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (load) {
          mlirDialectHandleLoadDialect(handle, context);
        }
      },
      nb::arg("context").none() = nb::none(), nb::arg("load") = true);
}
