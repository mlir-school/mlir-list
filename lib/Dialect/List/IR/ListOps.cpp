//===- ListOps.cpp - List dialect ops -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MyProject/Dialect/List/IR/ListOps.h"
#include "MyProject/Dialect/List/IR/ListDialect.h"
#include "MyProject/Dialect/List/IR/ListTypes.h"

#define GET_OP_CLASSES
#include "MyProject/Dialect/List/IR/ListOps.cpp.inc"

using namespace mlir;
using namespace list;

//===----------------------------------------------------------------------===//
// MapOpOp
//===----------------------------------------------------------------------===//

void MapOp::print(OpAsmPrinter &p) {
  p << " " << getList();
  p << " with ";
  p << "(" << getInductionVar() << " : ";
  p.printType(getInductionVar().getType());
  p << ") -> ";
  p.printType(getResult().getType().getElementType());
  p << " ";
  p.printRegion(getRegion(), /*printEntryBlockArgs=*/false);
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs*/{});
}

ParseResult MapOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();

  OpAsmParser::UnresolvedOperand inputList;
  OpAsmParser::Argument inductionVar;
  Type outputType;

  // Parse the input list
  if (parser.parseOperand(inputList))
    return failure();

  // Parse the "with" keyword
  if (parser.parseKeyword("with"))
    return failure();

  // Parse induction variable and type
  if (parser.parseLParen() ||
      parser.parseOperand(inductionVar.ssaName) ||
      parser.parseColonType(inductionVar.type) ||
      parser.parseRParen())
      return failure();

  // Parse output list scalar type
  if (parser.parseArrow() ||
      parser.parseType(outputType))
    return failure();

  // Parse the body region
  SmallVector<OpAsmParser::Argument> regionArgs = {inductionVar};
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, regionArgs))
    return failure();
  MapOp::ensureTerminator(*body, builder, result.location);

  // Parse optional attributes
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  
  // Resolve the input list type
  auto inputListType = builder.getType<ListType>(inductionVar.type);
  if (parser.resolveOperand(inputList, inputListType, result.operands))
    return failure();

  // Set the output list type
  auto outputListType = builder.getType<ListType>(outputType);
  result.types.push_back(outputListType);

  return success();
}

