//===- ListToStandard.h - Convert List to Standard dialect --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LISTPROJECT_CONVERSION_LISTTOSTANDARD_LISTTOSTANDARD_H
#define LISTPROJECT_CONVERSION_LISTTOSTANDARD_LISTTOSTANDARD_H

#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
class Location;
class OpBuilder;
class Pass;
class RewritePattern;
class RewritePatternSet;
class Value;
class ValueRange;

#define GEN_PASS_DECL_LOWERLISTPASS
#include "mlir/Conversion/Passes.h.inc"

/// Collect a set of patterns to convert from the List dialect to the Standard
/// dialect
REMOVE_ME!! a function declaration...why not !!!REMOVE_ME
void populateListToStdConversionPatterns(RewritePatternSet &patterns);

} // namespace mlir

#endif // LISTPROJECT_CONVERSION_LISTTOSTANDARD_LISTTOSTANDARD_H
