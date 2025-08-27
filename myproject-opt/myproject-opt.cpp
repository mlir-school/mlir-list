//===- myproject-opt.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "MyProject/Dialect/List/IR/ListDialect.h"
#include "MyProject/Dialect/List/Transforms/ListPasses.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::list::registerPasses();

  mlir::DialectRegistry registry;
  registerAllDialects(registry);
  registry.insert<mlir::list::ListDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "MyProject optimizer driver\n", registry));
}
