//===- ListToStandard.cpp - Lower list constructs to primitives -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file lowers list operation into standard dialects operations (mostly
// SCF and Tensor).
//
//===----------------------------------------------------------------------===//

#include "ListProject/Conversion/ListToStandard/ListToStandard.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#include "ListProject/Dialect/List/IR/ListOps.h"

#include <iostream>
namespace mlir {
#define GEN_PASS_DEF_LOWERLISTPASS
#include "ListProject/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::list;



namespace {
Type convertListType(Type type, int64_t size = -1) {
  if (auto listType = dyn_cast<ListType>(type)) {
    SmallVector<int64_t> shape = {size};
    return RankedTensorType::get(shape, listType.getElementType());
  }
  return type;
}

class ListRangeLowering : public OpConversionPattern<list::RangeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(list::RangeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto c0 = rewriter.create<arith::ConstantIndexOp>
      (op.getLoc(), 0).getResult();
    auto c1 = rewriter.create<arith::ConstantIndexOp>
      (op.getLoc(), 1).getResult();
    auto length = rewriter.create<arith::SubIOp>(op.getLoc(), op.getUpperBound(), op.getLowerBound()).getResult();
    auto lengthIndex = rewriter.create<arith::IndexCastOp>(op.getLoc(), rewriter.getIndexType(), length).getResult();
    auto tensorType = convertListType(op.getResult().getType());

    SmallVector<OpFoldResult> tensorShape = {lengthIndex};
    auto emptyTensor = rewriter.create<tensor::EmptyOp>(op.getLoc(), tensorShape,
        cast<ListType>(op.getResult().getType()).getElementType()).getResult();
    scf::ForOp forOp = scf::ForOp::create(rewriter, op.getLoc(), c0, lengthIndex, c1, emptyTensor);
    rewriter.setInsertionPointToStart(forOp.getBody());
    
    auto i = rewriter.create<arith::IndexCastOp>(op.getLoc(), rewriter.getI32Type(), forOp.getInductionVar()).getResult();
    auto value = rewriter.create<arith::AddIOp>(op.getLoc(), i, op.getLowerBound()).getResult();
    auto modified = rewriter.create<tensor::InsertOp>(op.getLoc(), value, forOp.getBody()->getArgument(1), forOp.getInductionVar()).getResult();

    auto yieldOp = rewriter.create<scf::YieldOp>(op.getLoc(), modified);
    rewriter.replaceOp(op, forOp);
    return success();
  }
};

class ListLengthLowering : public OpConversionPattern<list::LengthOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(list::LengthOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto c0 = rewriter.create<arith::ConstantIndexOp>
      (op.getLoc(), 0).getResult();
    auto dimSize = rewriter.create<tensor::DimOp>
      (op.getLoc(), op.getList(), c0).getResult();
    rewriter.replaceOp(op, dimSize);
    return success();
  }
};

class ListMapLowering : public OpConversionPattern<list::MapOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(list::MapOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto c0 = rewriter.create<arith::ConstantIndexOp>
      (op.getLoc(), 0).getResult();
    auto c1 = rewriter.create<arith::ConstantIndexOp>
      (op.getLoc(), 1).getResult();
    auto tensorType = convertListType(op.getList().getType());
    auto dimSize = rewriter.create<tensor::DimOp>
      (op.getLoc(), adaptor.getList(), c0).getResult();
    auto resultTensorType = convertListType(op.getResult().getType());

    SmallVector<OpFoldResult> resultTensorShape = {dimSize};
    auto emptyResult = rewriter.create<tensor::EmptyOp>(op.getLoc(), resultTensorShape,
        cast<ListType>(op.getResult().getType()).getElementType()).getResult();
    scf::ForOp forOp = scf::ForOp::create(rewriter, op.getLoc(), c0, dimSize, c1, emptyResult);

    rewriter.setInsertionPointToStart(forOp.getBody());
    auto x = rewriter.create<tensor::ExtractOp>(op.getLoc(), adaptor.getList(), forOp.getInductionVar()).getResult();
    
    IRMapping map;
    map.map(op.getBody().front().getArgument(0), x);
    for (Operation& op : op.getBody().front().getOperations())
      rewriter.clone(op, map);

    auto yieldOp = dyn_cast<list::YieldOp>(forOp.getBody()->back());
    assert(yieldOp);
    auto resultTensor = rewriter.create<tensor::InsertOp>(op.getLoc(), yieldOp.getValue(), forOp.getBody()->getArgument(1), forOp.getInductionVar()).getResult();
    auto newYieldOp = rewriter.create<scf::YieldOp>(op.getLoc(), resultTensor);
    rewriter.replaceOp(yieldOp, newYieldOp);
    rewriter.replaceOp(op, forOp);
    return success();
  }
};
} // namespace

void mlir::populateListToStdConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<
      ListRangeLowering,
      ListLengthLowering,
      ListMapLowering>(patterns.getContext());
}

namespace {
class LowerList : public impl::LowerListPassBase<LowerList> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateListToStdConversionPatterns(patterns);
    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect, tensor::TensorDialect,
                           scf::SCFDialect>();
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace
