#!/bin/bash

mkdir -p tmp/bin

cp -r ../../../../../third_party/llama.cpp_houmo/lib ./tmp/lib

DOCKER_REGISTRY="${DOCKER_REGISTRY:-registry.example.invalid}"
docker build -t "${DOCKER_REGISTRY}/application_software/hlie.cpp-base-xh1:v2.4.2-x86_64" .

rm -rf tmp
