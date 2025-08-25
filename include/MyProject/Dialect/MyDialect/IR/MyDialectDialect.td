//===- MyDialectDialect.td - MyDialect dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MYDIALECT_DIALECT
#define MYDIALECT_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// MyDialect dialect definition.
//===----------------------------------------------------------------------===//

def MyDialect_Dialect : Dialect {
    let name = "mydialect";
    let summary = "A mydialect out-of-tree MLIR dialect.";
    let description = [{
        This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
    }];
    let cppNamespace = "::mlir::mydialect";

    let useDefaultTypePrinterParser = 1;
    let extraClassDeclaration = [{
        void registerTypes();
    }];
}

//===----------------------------------------------------------------------===//
// Base mydialect operation definition.
//===----------------------------------------------------------------------===//

class MyDialect_Op<string mnemonic, list<Trait> traits = []> :
        Op<MyDialect_Dialect, mnemonic, traits>;

#endif // MYDIALECT_DIALECT
