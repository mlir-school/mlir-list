//===- MyDialectTypes.td - MyDialect dialect types -------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MYDIALECT_TYPES
#define MYDIALECT_TYPES

include "mlir/IR/AttrTypeBase.td"
include "MyProject/Dialect/MyDialect/IR/MyDialectDialect.td"

//===----------------------------------------------------------------------===//
// MyDialect type definitions
//===----------------------------------------------------------------------===//

class MyDialect_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<MyDialect_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def MyDialect_CustomType : MyDialect_Type<"Custom", "custom"> {
    let summary = "MyDialect custom type";
    let description = "Custom type in mydialect dialect";
    let parameters = (ins StringRefParameter<"the custom value">:$value);
    let assemblyFormat = "`<` $value `>`";
}

#endif // MYDIALECT_TYPES
