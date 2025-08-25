add_mlir_dialect_library(MLIRMyDialectTransforms
	MyDialectPasses.cpp

        ADDITIONAL_HEADER_DIRS
	${PROJECT_SOURCE_DIR}/include/MyProject/Dialect/MyDialect/Transforms

        DEPENDS
	MLIRMyDialectPassesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRPass
        )
