# Exercise 1: Your first operation

Welcome in **MLIR-List** project. This is an out-of-tree project which will allow you to define your custom dialect in the MLIR ecosystem.

## Project structure

Please take a few minutes to explore the project structure. Like many C++ projects, most of the code is split between the `include` dir for headers and `lib` for the sources. You will also find some [FileCheck](https://mlir.llvm.org/getting_started/TestingGuide/) based tests under `test`, and a `python` directory to allow you to create Python bindings for your dialect.

Once your have built the project by calling the `build.sh` script, you have access to a `listproject-opt` binary generated in `build/bin/` (please source `tosource.sh` to have it your path). This is a superset of the upstream `mlir-opt`, with you dialect and custom passes registered.

## Operation definition in ODS

For now, the project contains only one dialect, the `List` dialect. Following the standard project structure of MLIR projects, the include directory is at `./include/ListProject/Dialect/List` and the source directory is at `./lib/Dialect/List`.

One of the main things that a dialect bring are new type, attributes, and operations. These are located in the `IR` subdirectory. The common way to declare them is to use the [ODS](https://mlir.llvm.org/docs/DefiningDialects/Operations/) langage, derived from [TableGen](https://llvm.org/docs/TableGen/). The file format for `ODS` files is `.td`. After running tablegen scripts, it will generate both C++ headers (`.h.inc` files) and sources (`.cpp.inc` files).

For this first exercise, follow the error in the failing `.td` file and fix the declaration of the `list.foo` operation.

*Note: Don't forget to remove the TODO  ;)*
