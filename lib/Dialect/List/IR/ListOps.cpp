//===- ListOps.cpp - List dialect ops -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Block.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/ValueRange.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

#include "ListProject/Dialect/List/IR/ListOps.h"
#include "ListProject/Dialect/List/IR/ListDialect.h"
#include "ListProject/Dialect/List/IR/ListTypes.h"

#define GET_OP_CLASSES
#include "ListProject/Dialect/List/IR/ListOps.cpp.inc"

using namespace mlir;
using namespace list;

//===----------------------------------------------------------------------===//
// MapOp
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

LogicalResult MapOp::verify() {
  // Check that the type of yielded value is the same
  // as the element type of the result list
  auto resultElementType = this->getResult().getType().getElementType();
  auto yieldOp = dyn_cast<list::YieldOp>(this->getBody().front().back());
  assert(yieldOp);
  if (resultElementType != yieldOp.getValue().getType())
    return emitError() << "Element type of the result list does not match "
               << "the type of the yielded value: ("
               << resultElementType << " vs "
               << yieldOp.getValue().getType() << ")";
  return success();
}

//===----------------------------------------------------------------------===//
// LengthOp
//===----------------------------------------------------------------------===//

namespace {
struct EraseLengthOfRange : public OpRewritePattern<LengthOp> {
  using OpRewritePattern<LengthOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LengthOp lengthOp,
                                PatternRewriter &rewriter) const override {
    // Get the list
    auto list = lengthOp.getList();

    if (!list.getDefiningOp())
      return failure();

    if (!list.hasOneUse())
      return failure();

    if (auto rangeOp = dyn_cast<list::RangeOp>(*list.getDefiningOp())) {
	auto lowerBound = rangeOp.getLowerBound();
	auto upperBound = rangeOp.getUpperBound();
	auto length = rewriter.create<arith::SubIOp>(lengthOp.getLoc(), upperBound, lowerBound).getResult();
	rewriter.eraseOp(rangeOp);
	rewriter.replaceOp(lengthOp, length);
	return success();
    }

    return failure();
  }
};
} // namespace

void LengthOp::getCanonicalizationPatterns(RewritePatternSet &results,
						 MLIRContext *context) {
  results.add<EraseLengthOfRange>(context);
}
