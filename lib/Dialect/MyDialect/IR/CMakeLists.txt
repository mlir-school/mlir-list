add_mlir_dialect_library(MLIRMyDialect
	MyDialectTypes.cpp
	MyDialectDialect.cpp
	MyDialectOps.cpp

        ADDITIONAL_HEADER_DIRS
	${PROJECT_SOURCE_DIR}/include/MyProject/Dialect/MyDialect

        DEPENDS
	MLIRMyDialectOpsIncGen
	MLIRMyDialectPassesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        )
