set(LLVM_TARGET_DEFINITIONS ListPasses.td)
mlir_tablegen(ListPasses.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRListPassesIncGen)
