# MLIR-List: An out-of-tree MLIR dialect for List-Lang

Welcome in this project, where you will learn how to create your first dialect in the standard ODS/C++ way.
The session is organized in several exercises, each one in a separate Git branch.

## General information

### Installation

The session is a hand-on where you will actually modify an out-of-tree MLIR project (named *mlir-list*). To build it, you need an environment with prebuilt llvm-project:

#### Option 1: With Docker

*Note: To run docker without sudo once you have install it, please follow the post-installation instructions: https://docs.docker.com/engine/install/linux-postinstall/*

This is the simpliest option. You can download a prebuilt docker image by running:
```sh
docker pull electrikspace/mlir-tutorial
```
Then you can run the docker and work inside with:
```sh
docker run -it electrikspace/mlir-tutorial
```
If you want edit files locally and build with from the docker, we suggest you to clone the project and use the *in-docker* script, which basically run a command inside the docker env.
```sh
git clone https://github.com/mlir-school/mlir-list.git
chmod 777 mlir-list
./in-docker ./build.sh
```

#### Option 2: With prebuilt packages

If you want to work locally, a prebuilt llvm-project with MLIR is available as a Python package.

First, you need to install some dependancies and utils. The following commands have been tested on *Ubuntu24.10*:
```sh
sudo apt install python3 python3-pip python3-venv git clang lld
```
*Note: lld allows you to reduce the link time of mlir binaries over ld*

Create a Python virtual env (recommended):
```sh
python3 -m venv venv
source venv/bin/activate
```
Install some Python dependancies:
```sh
pip install lit pybind11 cmake ninja nanobind
```
Then install the prebuilt llvm-project with:
```sh
pip install --index-url https://gitlab.inria.fr/api/v4/groups/corse/-/packages/pypi/simple mlir-dev
```
Finally clone the session's project:
```sh
git clone https://github.com/mlir-school/mlir-list.git && cd mlir-list
```

### Build the project and run the tests

To build the project and run the tests:
```sh
./build.sh
```

If you want to use the generated `listproject-op` tool manually (requiered to run manual tests):
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

