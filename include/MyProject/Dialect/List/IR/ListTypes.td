//===- ListTypes.td - List dialect types -------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIST_TYPES
#define LIST_TYPES

include "mlir/IR/AttrTypeBase.td"
include "MyProject/Dialect/List/IR/ListDialect.td"

//===----------------------------------------------------------------------===//
// List type definitions
//===----------------------------------------------------------------------===//

class List_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<List_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def List_CustomType : List_Type<"Custom", "custom"> {
    let summary = "List custom type";
    let description = "Custom type in list dialect";
    let parameters = (ins StringRefParameter<"the custom value">:$value);
    let assemblyFormat = "`<` $value `>`";
}

#endif // LIST_TYPES
