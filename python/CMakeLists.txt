include(AddMLIRPython)

# Specifies that all MLIR packages are co-located under the `mlir_listproject`
# top level package (the API has been embedded in a relocatable way).
# TODO: Add an upstream cmake param for this vs having a global here.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=mlir_listproject.")


################################################################################
# Sources
################################################################################

declare_mlir_python_sources(ListProjectPythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT ListProjectPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_listproject"
  TD_FILE dialects/ListOps.td
  SOURCES
    dialects/list_pybind11.py
    dialects/list_nanobind.py
  DIALECT_NAME list)


declare_mlir_python_extension(ListProjectPythonSources.Pybind11Extension
  MODULE_NAME _listprojectDialectsPybind11
  ADD_TO_PARENT ListProjectPythonSources
  SOURCES
    ListProjectExtensionPybind11.cpp
  EMBED_CAPI_LINK_LIBS
    ListProjectCAPI
  PYTHON_BINDINGS_LIBRARY pybind11
)

declare_mlir_python_extension(ListProjectPythonSources.NanobindExtension
  MODULE_NAME _listprojectDialectsNanobind
  ADD_TO_PARENT ListProjectPythonSources
  SOURCES
    ListProjectExtensionNanobind.cpp
  EMBED_CAPI_LINK_LIBS
    ListProjectCAPI
  PYTHON_BINDINGS_LIBRARY nanobind
)


################################################################################
# Common CAPI
################################################################################

add_mlir_python_common_capi_library(ListProjectPythonCAPI
  INSTALL_COMPONENT ListProjectPythonModules
  INSTALL_DESTINATION python_packages/listproject/mlir_listproject/_mlir_libs
  OUTPUT_DIRECTORY "${MLIR_BINARY_DIR}/python_packages/listproject/mlir_listproject/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    ListProjectPythonSources
    # TODO: Remove this in favor of showing fine grained registration once
    # available.
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources.Core
)

################################################################################
# Instantiation of all Python modules
################################################################################

add_mlir_python_modules(ListProjectPythonModules
  ROOT_PREFIX "${MLIR_BINARY_DIR}/python_packages/listproject/mlir_listproject"
  INSTALL_PREFIX "python_packages/listproject/mlir_listproject"
  DECLARED_SOURCES
    ListProjectPythonSources
    # TODO: Remove this in favor of showing fine grained registration once
    # available.
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources
  COMMON_CAPI_LINK_LIBS
    ListProjectPythonCAPI
  )
