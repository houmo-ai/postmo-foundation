#!/bin/bash

mkdir -p tmp/config
mkdir -p tmp/lib
mkdir -p tmp/bin
cp -r ../../../../../cmake-build-release/bin/libhlie-*.so ./tmp/lib/
cp -r ../../../../../cmake-build-release/bin/houmo-llm-hi-cpp-linux ./tmp/bin/HLIECpp
cp -r ../../../../../config/linux/sys.cfg ./tmp/config/sys.cfg
cp -r ../../../../../config/config_schema.json ./tmp/config/config_schema.json
cp -r ../../../../../third_party/pdfium/lib/*.so ./tmp/lib/ 2>/dev/null || true
cp -r ../../../../../third_party/pdfium/bin/icudtl.dat ./tmp/bin/ 2>/dev/null || true

docker build -t harbor.houmo.ai/application_software/hlie.cpp-xh2-20250910:v0.3.0-x86_64 .

rm -rf tmp
