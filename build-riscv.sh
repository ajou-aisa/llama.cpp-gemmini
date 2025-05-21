#!/bin/bash

target=riscv
if [ "$1" = "static" ]; then
  target=riscv-$1
fi
mkdir -p "build-$target"
cd "build-$target"
cmake -DCMAKE_TOOLCHAIN_FILE=../cmake/$target.cmake ..
make llama-cli -j
