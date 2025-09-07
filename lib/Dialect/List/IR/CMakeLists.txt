add_mlir_dialect_library(MLIRListDialect
	ListTypes.cpp
	ListDialect.cpp
	ListOps.cpp

        ADDITIONAL_HEADER_DIRS
	${PROJECT_SOURCE_DIR}/include/ListProject/Dialect/List

        DEPENDS
	MLIRListOpsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        )
