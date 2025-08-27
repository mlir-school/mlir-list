include(AddMLIRPython)

# Specifies that all MLIR packages are co-located under the `mlir_myproject`
# top level package (the API has been embedded in a relocatable way).
# TODO: Add an upstream cmake param for this vs having a global here.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=mlir_myproject.")


################################################################################
# Sources
################################################################################

declare_mlir_python_sources(MyProjectPythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT MyProjectPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_myproject"
  TD_FILE dialects/ListOps.td
  SOURCES
    dialects/list_pybind11.py
    dialects/list_nanobind.py
  DIALECT_NAME list)


declare_mlir_python_extension(MyProjectPythonSources.Pybind11Extension
  MODULE_NAME _myprojectDialectsPybind11
  ADD_TO_PARENT MyProjectPythonSources
  SOURCES
    MyProjectExtensionPybind11.cpp
  EMBED_CAPI_LINK_LIBS
    MyProjectCAPI
  PYTHON_BINDINGS_LIBRARY pybind11
)

declare_mlir_python_extension(MyProjectPythonSources.NanobindExtension
  MODULE_NAME _myprojectDialectsNanobind
  ADD_TO_PARENT MyProjectPythonSources
  SOURCES
    MyProjectExtensionNanobind.cpp
  EMBED_CAPI_LINK_LIBS
    MyProjectCAPI
  PYTHON_BINDINGS_LIBRARY nanobind
)


################################################################################
# Common CAPI
################################################################################

add_mlir_python_common_capi_library(MyProjectPythonCAPI
  INSTALL_COMPONENT MyProjectPythonModules
  INSTALL_DESTINATION python_packages/myproject/mlir_myproject/_mlir_libs
  OUTPUT_DIRECTORY "${MLIR_BINARY_DIR}/python_packages/myproject/mlir_myproject/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    MyProjectPythonSources
    # TODO: Remove this in favor of showing fine grained registration once
    # available.
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources.Core
)

################################################################################
# Instantiation of all Python modules
################################################################################

add_mlir_python_modules(MyProjectPythonModules
  ROOT_PREFIX "${MLIR_BINARY_DIR}/python_packages/myproject/mlir_myproject"
  INSTALL_PREFIX "python_packages/myproject/mlir_myproject"
  DECLARED_SOURCES
    MyProjectPythonSources
    # TODO: Remove this in favor of showing fine grained registration once
    # available.
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources
  COMMON_CAPI_LINK_LIBS
    MyProjectPythonCAPI
  )
