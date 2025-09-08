
REMOVE_ME!!! Sometimes you need to write (copy paste) cmake stuff !!!REMOVE_ME

add_mlir_dialect_library(MLIRListTransforms
	ListPasses.cpp

        ADDITIONAL_HEADER_DIRS
	${PROJECT_SOURCE_DIR}/include/ListProject/Dialect/List/Transforms

        DEPENDS
	MLIRListPassesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRPass
        )
