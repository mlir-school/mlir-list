include(AddMLIRPython)

# Specifies that all MLIR packages are co-located under the `mlir_listproject`
# top level package (the API has been embedded in a relocatable way).
# TODO: Add an upstream cmake param for this vs having a global here.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=${MLIR_PYTHON_PACKAGE_PREFIX}.")

################################################################################
# Sources
################################################################################

declare_mlir_python_sources(ListProjectPythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT ListProjectPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_listproject"
  TD_FILE dialects/ListOps.td
  SOURCES
    dialects/list_nanobind.py
  DIALECT_NAME list)

declare_mlir_python_extension(ListProjectPythonSources.NanobindExtension
  MODULE_NAME _listprojectDialectsNanobind
  ADD_TO_PARENT ListProjectPythonSources
  SOURCES
    ListProjectExtensionNanobind.cpp
  EMBED_CAPI_LINK_LIBS
    ListProjectCAPI
  PYTHON_BINDINGS_LIBRARY nanobind
  GENERATE_TYPE_STUBS
)


################################################################################
# Common CAPI
################################################################################

add_mlir_python_common_capi_library(ListProjectPythonCAPI
  INSTALL_COMPONENT ListProjectPythonModules
  INSTALL_DESTINATION "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_mlir_libs"
  OUTPUT_DIRECTORY "${MLIR_BINARY_DIR}/${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    ListProjectPythonSources
    # TODO: Remove this in favor of showing fine grained registration once
    # available.
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources.Core
    MLIRPythonSources.Dialects.builtin
    MLIRPythonSources.Dialects.arith
    MLIRPythonSources.Dialects.memref
    # Add other dialects here, or import all with MLIRPythonSources.Dialects
)

################################################################################
# Instantiation of all Python modules
################################################################################

add_mlir_python_modules(ListProjectPythonModules
  ROOT_PREFIX "${MLIR_BINARY_DIR}/${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}"
  INSTALL_PREFIX "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}"
  DECLARED_SOURCES
    ListProjectPythonSources
    # TODO: Remove this in favor of showing fine grained registration once
    # available.
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources.Core
    MLIRPythonSources.Dialects.builtin
    MLIRPythonSources.Dialects.memref
    MLIRPythonSources.Dialects.arith
    # Add other dialects here, or import all with MLIRPythonSources.Dialects
  COMMON_CAPI_LINK_LIBS
    ListProjectPythonCAPI
  )
