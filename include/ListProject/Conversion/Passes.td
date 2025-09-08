//===-- Passes.td - Conversion pass definition file --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LISTPROJECT_CONVERSION_PASSES
#define LISTPROJECT_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// LowerList
//===----------------------------------------------------------------------===//

REMOVE_ME!! Declare you conversion pass !!!REMOVE_ME
def LowerListPass : Pass<"lower-list"> {
  let summary = "Lower List dialect into scf and tensor operations";
  let description = [{
    Lower List dialect into scf and tensori operations
  }];
  let dependentDialects = ["scf::SCFDialect", "tensor::TensorDialect"];
}

#endif // LISTPROJECT_CONVERSION_PASSES
