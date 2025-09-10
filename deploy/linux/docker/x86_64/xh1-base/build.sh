#!/bin/bash

mkdir -p tmp/bin

cp -r ../../../../../third_party/llama.cpp_houmo/lib ./tmp/lib

docker build -t harbor.houmo.ai/application_software/hlie.cpp-base-xh1:v2.4.2-x86_64 .

rm -rf tmp