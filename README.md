# Define And Lower Your New Dialect

## Abstract

In this hands-on session, you will learn how to create the "List" dialect, introduced in the first lesson. You will practice in an out-of-tree MLIR project and develop your dialect in the standard way, without using xDSL or IRDL. This will give you an opportunity to apply the concepts learned in the previous lessons.

The session is organized into multiple exercises, each exploring a different aspect of dialect creation. Every exercise comes with a README, hints, and links to the portion of the official MLIR documentation. You can progress in the exercises either with the teachers or at your own pace.

The goal is to provide you with both the knowledge and a project template to help you create your own dialect for your ideas.

## Installation

The session is a hand-on where you will actually modify an out-of-tree MLIR project (named *mlir-list*). To build it, you need an environment with prebuilt llvm-project:

### Option 1: With Docker

*Note: To run docker without sudo once you have install it, please follow the post-installation instructions: https://docs.docker.com/engine/install/linux-postinstall/*

This is the simpliest option. You can download a prebuilt docker image by running:
```
docker pull electrikspace/mlir-tutorial:v2
```
Then you can run the docker and work inside with:
```
docker run -it electrikspace/mlir-tutorial:v2
```
If you want edit files locally and build with from the docker, we suggest you to clone the project and use the *in-docker* script, which basically run a command inside the docker env.
```
git clone https://github.com/mlir-school/mlir-list.git
chmod 777 mlir-list
cd mlir-list
./in-docker ./build.sh
```

### Option 2: With prebuilt packages

If you want to work locally, a prebuilt llvm-project with MLIR is available as a Python package.

First, you need to install some dependancies and utils. The following commands have been tested on *Ubuntu24.10*:
```
sudo apt install python3 python3-pip python3-venv git clang lld
```
*Note: lld allows you to reduce the link time of mlir binaries over ld*

Create a Python virtual env (recommended):
```
python3 -m venv venv
source venv/bin/activate
```
Install some Python dependancies:
```
pip install lit pybind11 cmake ninja nanobind
```
Then install the prebuilt llvm-project with:
```
pip install --index-url https://gitlab.inria.fr/api/v4/groups/corse/-/packages/pypi/simple mlir-dev
```
Finally clone the session's project:
```
git clone https://github.com/mlir-school/mlir-list.git && cd mlir-list
```

### Test your installation

To test your installation, you can run the build script in the root directory of *mlir-list*:
```
./build.sh
```
If everything goes well, then your are ready! 

f you want to use the generated `listproject-op` tool manually (requiered to run manual tests):
```sh
source tosource.sh
```

All tests are in the `./test` directory. If you to test a specific test:
```sh
lit build/test/List/<test-name>.mlir
```

*Don't hesitate to take a look at the installation scripts. The Dockerfile used to build the image is located here: https://github.com/ElectrikSpace/mlir-tutorial-docker*

### List all exercises

```sh
git branch -a
```

### Start an exercise
```sh
git checkout ex<number>
```

Note: If you have already work on a previous exercise, you can save your work:
```
git add -u && git commit -m "My super work"
```

Then, follow the instruction written in the **README.md** file.

### List dialect: Remainder

#### `list.range`
Create a list given a lower bound and an upper bound.
Example:
```mlir
%c0 = arith.constant 0 : i32
%c5 = arith.constant 5 : i32
%int_list = list.range %c0 to %c5 : !list.list<i32>
```

#### `list.length`
Get the length of a list.
Example:
```mlir
%length = list.length %int_list : !list.list<i32> -> i32
```

#### `list.map`
Map a function to each element of a list, and get a new list.
Example:
```mlir
%new_list = list.map %int_list with (%x : i32) -> i1 {
    %c2 = arith.constant 2 : i32
    %x_ult_c2 = arith.cmpi ult, %x, %c2 : i32
list.yield %x_ult_c2 : i1
```

#### `list.yield`
Terminator for `list.map`

### Exercise 1: Your first operation

Welcome in **MLIR-List** project. This is an out-of-tree project which will allow you to define your custom dialect in the MLIR ecosystem.

#### Project structure

Please take a few minutes to explore the project structure. Like many C++ projects, most of the code is split between the `include` dir for headers and `lib` for the sources. You will also find some [FileCheck](https://mlir.llvm.org/getting_started/TestingGuide/) based tests under `test`, and a `python` directory to allow you to create Python bindings for your dialect.

Once your have built the project by calling the `build.sh` script, you have access to a `listproject-opt` binary generated in `build/bin/` (please source `tosource.sh` to have it your path). This is a superset of the upstream `mlir-opt`, with you dialect and custom passes registered.

#### Operation definition in ODS

For now, the project contains only one dialect, the `List` dialect. Following the standard project structure of MLIR projects, the include directory is at `./include/ListProject/Dialect/List` and the source directory is at `./lib/Dialect/List`.

One of the main things that a dialect bring are new type, attributes, and operations. These are located in the `IR` subdirectory. The common way to declare them is to use the [ODS](https://mlir.llvm.org/docs/DefiningDialects/Operations/) langage, derived from [TableGen](https://llvm.org/docs/TableGen/). The file format for `ODS` files is `.td`. After running tablegen scripts, it will generate both C++ headers (`.h.inc` files) and sources (`.cpp.inc` files).

For this first exercise, follow the error in the failing `.td` file and fix the declaration of the `list.foo` operation.

*Note: Don't forget to remove the TODO  ;)*

### Exercise 2: A type: `!list.list<>`

Before you create operations operating on lists, you will need to define a list type. Type definition can be done in `ODS`. Follow the error and fix the definition of the `ListType`.

**Clues**:

- ElementType is built by constraints: https://mlir.llvm.org/docs/DefiningDialects/Constraints/*
```td
// Example: Element type constraint for VectorType
def Builtin_VectorTypeElementType : AnyTypeOf<[AnyInteger, Index, AnyFloat]>;
```

- ListType is full type defined in ODS: (https://mlir.llvm.org/docs/DefiningDialects/AttributesAndTypes/#adding-a-new-attribute-or-type-definition)
```td
// Example from standalone MLIR example
def Standalone_CustomType : Standalone_Type<"Custom", "custom"> {
    let summary = "Standalone custom type";
    let description = "Custom type in standalone dialect";
    let parameters = (ins StringRefParameter<"the custom value">:$value);
    let assemblyFormat = "`<` $value `>`";
}
```

*Note: Don't forget to remove the TODO ;)*

### Exercise 3: Some operations: `list.range`, `list.map`, and `list.yield`

Now it's time to create your first real operation of the List dialect. In this exercise, you will learn how to write these operations in [ODS](https://mlir.llvm.org/docs/DefiningDialects/Operations/).

Follow the errors to fix the code. You **only need to replace the TODO!!!TODO** by your code, and **delete the REMOVE_ME!!!...!!!REMOVE_ME**. You should be able to complete the exercise by loop at other operations around, bu you can also read [this chapter from the Toy tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-2/) or the [ODS](https://mlir.llvm.org/docs/DefiningDialects/Operations/#operation-definition) documentation.

*Note: Don't forget to remove the TODO and REMOVE_ME ;)*

### Exercise 4: A verifier for `list.map`

Huston, we have a problem. It seems that you can write a `list.map` that returns a `!list.list<i32>` but yield a `i1` for each element. You need to prevent this!

Operation definition allows you to add a custom verifier in C++. This may solve the issue.

Follow the errors to fix the verifier of the `list.map`.

**Clues**:
- Documentation on class [Value](https://mlir.llvm.org/doxygen/classmlir_1_1Value.html)
- Given the ODS Definition of Types and ops you did before, getters are automatically generated for params/arguments/results:
```cpp
// Example: Access $tensor with .getTensor() 
let parameters = (ins "MemRefType":$tensor);
// Example: Access $input with .getInput()
let arguments = (ins I32:$input);
```
- (not used normally: Documentation on class [Operation](https://mlir.llvm.org/doxygen/classmlir_1_1Operation.html))

*Note: Don't forget to remove the TODO and REMOVE_ME ;)*

### Exercise 5: Add `list.length` with a canonicalization pattern

Let's take this example:
```mlir
%l = list.range %c0 to %c10 : !list.list<i32>
%length = list.length %l : !list.list<i32> -> i32
```
Maybe there is a way to simplify this structure, a length of a range, to something like this:
```mlir
%length = arith.subi %c10, %c0 : i32
```

To do this in MLIR, you can implement a canonicalization pattern on the `list.length` operation, in C++. The `--canonicalize` pass is the responsible to call all the patterns.

Follow the error and fix the canonicalization pattern for `list.length`.

**Clues**:
- Documentation on class [Value](https://mlir.llvm.org/doxygen/classmlir_1_1Value.html)
- Documentation on class [Operation](https://mlir.llvm.org/doxygen/classmlir_1_1Operation.html))
- llvm [`dyn_cast`](https://llvm.org/docs/ProgrammersManual.html#the-isa-cast-and-dyn-cast-templates)

*Note: Don't forget to remove the TODO and REMOVE_ME ;)*

### Exercise 6: A transform Pass

Now you have all the basics list operation. So you can begin to manipulate the IR in passes.

There are two types of passes: Transformations that operate at a giving dialect, and Conversion/lowering passes that convert from a dialect to anothers.

For this exercise, you will write a (dummy) transformation that removes `list.foo` ops only if it contains an atttribute named `useless`. In this case, you have to replace the result by the input. 

**Clues**:
- Documentation on class [Operation](https://mlir.llvm.org/doxygen/classmlir_1_1Operation.html))

*Note: [to Understand the walker used in the pass](https://mlir.llvm.org/docs/Tutorials/UnderstandingTheIRStructure/)*

*Note: Don't forget to remove the TODO and REMOVE_ME ;)*

### Exercise 7: A lowering Pass

*[many transformations later]*

You are now ready to lower your super optimized list IR into standard MLIR dialects, to in fine generate an executable.

A lowering pass can be seen as a Conversion from a dialect to one or many others. Conversion passes have some nice properties that you can exlore [here](https://mlir.llvm.org/docs/DialectConversion/). It is usually implemented with a special [`ConversionPattern`](https://mlir.llvm.org/docs/DialectConversion/#conversion-patterns). Unlike more general `RewritePattern`, it can handle [type conversion](https://mlir.llvm.org/docs/DialectConversion/#type-conversion).

Fix the conversion pass from list to SCF/Arith/Tensor dialects.

**Clues**:
- Documentation on class [ConversionPatternRewriter](https://mlir.llvm.org/doxygen/classmlir_1_1ConversionPatternRewriter.html). (Don't forget to also explore inherited methods)
- [`tensor.dim`](https://mlir.llvm.org/docs/Dialects/TensorOps/#tensordim-tensordimop)

*Note: The list conversion pass does not require a special [`TypeConverter`](https://mlir.llvm.org/docs/DialectConversion/#type-converter), because there is not remaining operation operating on `!list.list<>` once all conversion pattern have been applied.*

*Note: Don't forget to remove the TODO and REMOVE_ME ;)*

### Exercise 8: Python bindings

Python bindings for free!!! It sounds great, doesn't ?

MLIR exposes a [C API](https://mlir.llvm.org/docs/CAPI/) to manipulate the IR from C programs. Then, [Python bindings](https://mlir.llvm.org/docs/Bindings/Python/) can be derived from this, using nanobind

*Note: Please add `-DMLIR_ENABLE_BINDINGS_PYTHON=ON` cmake option to enable Python bindings when building `llvm-project` or your out-of-tree project.*

Aside from the dialect registration, you can have most of the bindings for free. But if you have created new types or attributes, you will need to add [CAPI](https://mlir.llvm.org/docs/CAPI/#extensions-for-dialect-attributes-and-types) and [Python](https://mlir.llvm.org/docs/Bindings/Python/#attributes-and-types-2) bindings by yourself.

For this last exercise, feel free to explore the CAPI under `include/ListProject-c/` and `lib/CAPI/`. The extensions for Python ar under `python/`, which contains Nanobind code in C++ and Python code under `python/mlir_listproject/`.

Follow the errors to understand how to create bindings for the `list.list<>` type. 

*Note: Don't forget to remove the TODO and REMOVE_ME ;)*

### Beyond

This project is derived from the standalone example in `llvm-project/mlir/examples/standalone/`. Feel free to reuse and adapt this code or the standalone example to create your own dialect.
