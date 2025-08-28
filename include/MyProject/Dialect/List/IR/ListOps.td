//===- ListOps.td - List dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIST_OPS
#define LIST_OPS

include "MyProject/Dialect/List/IR/ListTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def List_FooOp : List_Op<"foo", [Pure,
                                             SameOperandsAndResultType]> {
    let summary = "Illustrates how to define an operation.";
    let description = [{
        The `list.foo` operation illustrates how to define a new
        operation in a dialect. It uses an operation trait to declare that it
        has no side effects.

        This operation takes an integer argument and returns an integer.

        Example:

        ```mlir
        %0 = arith.constant 2 : i32
        // Apply the foo operation to %0
        %1 = list.foo %0 : i32
        ```
    }];

    let arguments = (ins I32:$input);
    let results = (outs I32:$res);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

def List_RangeOp : List_Op<"range", [Pure]> {
    let summary = "Create a list from a range.";
    let description = [{
	TODO
    }];

    let arguments = (ins I32:$lowerBound,
			 I32:$upperBound);
    let results = (outs AnyTypeOf<[List_ListType]>:$result);
    // This version is better, but the whole type is not printed
    //let results = (outs List_ListType:$result);

    let assemblyFormat = [{
        $lowerBound `to` $upperBound attr-dict `:` type($result)
    }];
}

def List_LengthOp : List_Op<"length", [Pure]> {
    let summary = "Get the length of a list.";
    let description = [{
	TODO
    }];

    let arguments = (ins AnyTypeOf<[List_ListType]>:$list);
    // This version is better, but the whole type is not printed
    //let arguments = (ins List_ListType:$list);
    let results = (outs I32:$result);

    let assemblyFormat = [{
        $list attr-dict `:` type($list) `->` type($result)
    }];
    let hasCanonicalizer = 1;
}

def List_MapOp : List_Op<"map", [AffineScope, SingleBlockImplicitTerminator<"YieldOp">]> {
    let summary = "A yield op for list map.";
    let description = [{
	TODO
    }];

    let arguments = (ins List_ListType:$list);
    let results = (outs List_ListType:$result);
    
    let regions = (region SizedRegion<1>:$body);

    let extraClassDeclaration = [{
	// Get induction Variable
	Value getInductionVar() { return getBody().getArgument(0); }
    }];
    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;
}

def List_YieldOp : List_Op<"yield", [HasParent<"MapOp">, Terminator]> {
    let summary = "A yield op for list map.";
    let description = [{
	TODO
    }];

    let arguments = (ins Variadic<List_ElementType>:$values);
    let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];

    let assemblyFormat = [{
        attr-dict ($values^ `:` type($values))?
    }];
    // hasVerifier = 1;
}

#endif // LIST_OPS
