add_mlir_public_c_api_library(MyProjectCAPI
  Dialects.cpp
  LINK_LIBS PUBLIC
  MLIRList
)
