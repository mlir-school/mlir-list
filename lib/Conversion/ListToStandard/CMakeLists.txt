add_mlir_conversion_library(MLIRListToStandard
  ListToStandard.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/MyProject/Conversion/ListToStandard

  DEPENDS
  MLIRListProjectConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRListDialect

  MLIRArithDialect
  MLIRIR
  MLIRTensorDialect
  MLIRPass
  MLIRSCFDialect
  MLIRTransforms
  )
