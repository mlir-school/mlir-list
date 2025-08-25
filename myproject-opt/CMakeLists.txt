set(LIBS
    MLIRArithDialect
    MLIROptLib
    MLIRRegisterAllDialects
    MLIRRegisterAllPasses
    MLIRMyDialect
    MLIRMyDialectTransforms
  )
add_llvm_executable(myproject-opt myproject-opt.cpp)

llvm_update_compile_flags(myproject-opt)
target_link_libraries(myproject-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(myproject-opt)
