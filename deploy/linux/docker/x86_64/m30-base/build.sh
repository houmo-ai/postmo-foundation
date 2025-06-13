#!/bin/bash

mkdir -p tmp/bin
cp -r ../../../../../src/libtorch/ ./tmp/libtorch
cp -r ../../../../../src/houmo-tcim-runtime/ ./tmp/houmo-tcim-runtime
cp ../llama_deps/* ./tmp/bin
cp /usr/local/houmo-sdk/hal/lib/* ./tmp/bin/

docker build -t harbor.houmo.ai/application_software/hi.cpp-base-m30:v2.3.0-x86_64 .

rm -rf tmp