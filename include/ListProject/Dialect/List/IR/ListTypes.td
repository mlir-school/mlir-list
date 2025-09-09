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
include "ListProject/Dialect/List/IR/ListDialect.td"

//===----------------------------------------------------------------------===//
// List type definitions
//===----------------------------------------------------------------------===//

class List_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<List_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def List_ElementType: AnyTypeOf<[AnyInteger]>;

def List_ListType : List_Type<"List", "list"> {
    let summary = "List list type";
    let description = "List type in list dialect";
    let parameters = (ins List_ElementType:$element_type);
    let assemblyFormat = "`<` $element_type `>`";
}

#endif // LIST_TYPES
