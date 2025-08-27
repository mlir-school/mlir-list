add_mlir_dialect_library(MLIRListTransforms
	ListPasses.cpp

        ADDITIONAL_HEADER_DIRS
	${PROJECT_SOURCE_DIR}/include/MyProject/Dialect/List/Transforms

        DEPENDS
	MLIRListPassesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRPass
        )
