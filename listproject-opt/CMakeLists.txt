set(LIBS
    MLIRArithDialect
    MLIROptLib
    MLIRRegisterAllDialects
    MLIRRegisterAllPasses
    MLIRListDialect
    MLIRListTransforms
  )
add_llvm_executable(listproject-opt listproject-opt.cpp)

llvm_update_compile_flags(listproject-opt)
target_link_libraries(listproject-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(listproject-opt)
