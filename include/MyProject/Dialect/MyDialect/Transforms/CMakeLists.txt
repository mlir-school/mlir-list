set(LLVM_TARGET_DEFINITIONS MyDialectPasses.td)
mlir_tablegen(MyDialectPasses.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRMyDialectPassesIncGen)
