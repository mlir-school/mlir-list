get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
    ${conversion_libs}
    MLIRArithDialect
    MLIROptLib
    MLIRRegisterAllDialects
    MLIRRegisterAllPasses
    MLIRListDialect
  )
add_llvm_executable(listproject-opt listproject-opt.cpp)

llvm_update_compile_flags(listproject-opt)
target_link_libraries(listproject-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(listproject-opt)
