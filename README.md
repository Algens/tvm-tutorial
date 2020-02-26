## What is this about?
Very truly simple tutorial for TVM(with NNVM for a certain project)

## How to install LLVM on Linux
The project I'm working on uses NNVM compiler with TVM, so LLVM must be installed before TVM.
Please visit http://releases.llvm.org/download.html and download pre-built binary of version 9.0.0, considering your OS version.


## How to install TVM on Linux
Please see https://docs.tvm.ai/install/from_source.html with this tutorial.

1. Clone TVM repository with `--recursive` option
  ```bash
  git clone --recursive https://github.com/apache/incubator-tvm tvm
  ```

2. Install minimum building requirements
  ```bash
  sudo apt-get update
  sudo apt-get install -y python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev
  ```

3. Make build directory and copy config.cmake
  ```bash
  mkdir build
  cp cmake/config.cmake build
  ```

4. Insert following line in config.cmake in build directory
  ```bash
  set(USE_LLVM /path/to/your/llvm/bin/llvm-config)
  ```

5. Now build TVM
  ```bash
  cd build
  cmake ..
  make -j4
  ```

## Install TVM Python packages
Please insert following lines in `~/.bashrc`
  ```bash
  export TVM_HOME=/path/to/tvm
  export PYTHONPATH=$TVM_HOME/python:$TVM_HOME/topi/python:$TVM_HOME/nnvm/python:${PYTHONPATH}
  ```
