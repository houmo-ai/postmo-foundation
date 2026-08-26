#!/bin/bash

mkdir -p tmp/bin

cp -r ../../../../../third_party/llama.cpp_houmo/lib ./tmp/lib
cp -r ../../../../../third_party/sd.cpp/lib/libstable-diffusion.so ./tmp/lib/libstable-diffusion.so

docker build -t harbor.houmo.ai/application_software/hlie.cpp-base-xh2:v0.3.0-x86_64 .

rm -rf tmp