//===- Passes.h - Conversion Pass Construction and Registration -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LISTPROJECT_CONVERSION_PASSES_H
#define LISTPROJECT_CONVERSION_PASSES_H

#include "ListProject/Conversion/ListToStandard/ListToStandard.h"

namespace mlir {

/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "ListProject/Conversion/Passes.h.inc"

} // namespace mlir

#endif // LISTPROJECT_CONVERSION_PASSES_H
