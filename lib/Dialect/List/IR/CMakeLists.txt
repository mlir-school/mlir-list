add_mlir_dialect_library(MLIRList
	ListTypes.cpp
	ListDialect.cpp
	ListOps.cpp

        ADDITIONAL_HEADER_DIRS
	${PROJECT_SOURCE_DIR}/include/ListProject/Dialect/List

        DEPENDS
	MLIRListOpsIncGen
	MLIRListPassesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        )
