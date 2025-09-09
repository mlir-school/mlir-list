//===- ListPsss.td - List dialect passes -------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIST_PASS
#define LIST_PASS

include "mlir/Pass/PassBase.td"

def ListRemoveSomeFoo: Pass<"list-remove-some-foo", "::mlir::ModuleOp"> {
  let summary = "Remove some `foo` operations.";
  let description = [{
    Identify the `list.foo` operation with an `useless` attribute, and
    remove them.
  }];
}

#endif // LIST_PASS
