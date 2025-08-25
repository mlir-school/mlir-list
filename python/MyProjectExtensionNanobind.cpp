//===- MyProjectExtension.cpp - Extension module --------------------------===//
//
// This is the nanobind version of the example module. There is also a pybind11
// example in MyProjectExtensionPybind11.cpp.
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MyProject-c/Dialects.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

namespace nb = nanobind;

NB_MODULE(_myprojectDialectsNanobind, m) {
  //===--------------------------------------------------------------------===//
  // mydialect dialect
  //===--------------------------------------------------------------------===//
  auto mydialectM = m.def_submodule("mydialect");

  mydialectM.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle handle = mlirGetDialectHandle__mydialect__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (load) {
          mlirDialectHandleLoadDialect(handle, context);
        }
      },
      nb::arg("context").none() = nb::none(), nb::arg("load") = true);
}
