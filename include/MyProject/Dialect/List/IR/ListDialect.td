//===- ListDialect.td - List dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIST_DIALECT
#define LIST_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// List dialect definition.
//===----------------------------------------------------------------------===//

def List_Dialect : Dialect {
    let name = "list";
    let summary = "A list out-of-tree MLIR dialect.";
    let description = [{
        This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
    }];
    let cppNamespace = "::mlir::list";

    let useDefaultTypePrinterParser = 1;
    let extraClassDeclaration = [{
        void registerTypes();
    }];
}

//===----------------------------------------------------------------------===//
// Base list operation definition.
//===----------------------------------------------------------------------===//

class List_Op<string mnemonic, list<Trait> traits = []> :
        Op<List_Dialect, mnemonic, traits>;

#endif // LIST_DIALECT
