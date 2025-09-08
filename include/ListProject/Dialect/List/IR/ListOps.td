//===- ListOps.td - List dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIST_OPS
#define LIST_OPS

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "ListProject/Dialect/List/IR/ListDialect.td"
include "ListProject/Dialect/List/IR/ListTypes.td"

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
        The `list.range` operation creates a list given a lower bound and
        an upper bound.

        Example:

        ```mlir
        %c0 = arith.constant 0 : i32
        %c10 = arith.constant 10 : i32
        %list = list.range %c0 to %c10 : !list.list<i32>
        ```
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

def List_MapOp : List_Op<"map", [AffineScope, SingleBlockImplicitTerminator<"YieldOp">]> {
    let summary = "A yield op for list map.";
    let description = [{
	    The `list.map` operation applies the inner block to each element of
        a list.

        Example:

        ```
        %pred_list = list.map %list with (%x : i32) -> i1 {
            %c42 = arith.constant 2 : i32
            %pred = arith.cmpi ult, %x, %c42 : i32
            list.yield %pred : i1
        }
        ```
    }];

    let arguments = (ins List_ListType:$list);
    let results = (outs List_ListType:$result);
    
    let regions = (region SizedRegion<1>:$body);

    let extraClassDeclaration = [{
	    // Get induction Variable
	    Value getInductionVar() { return getBody().getArgument(0); }
    }];
    let hasCustomAssemblyFormat = 1;

    REMOVE_ME!! Declare a custom verifier with this !!REMOVE_ME 
    let hasVerifier = 1;
}

def List_YieldOp : List_Op<"yield", [HasParent<"MapOp">, Terminator]> {
    let summary = "A yield op for list map.";
    let description = [{
	    The `list.yield` operation is a terminator for a `list.map` operation.
    }];

    let arguments = (ins List_ElementType:$value);
    let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];

    let assemblyFormat = [{
        $value attr-dict `:` type($value)
    }];
}

#endif // LIST_OPS
