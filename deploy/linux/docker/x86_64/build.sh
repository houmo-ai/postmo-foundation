#!/bin/bash

mkdir tmp
cp -r ../../../../cmake-build-release/bin/* ./tmp/

docker build -t harbor.houmo.ai/application_software/hi.cpp-m30:v2.3.0-x86_64 .

rm -rf tmp