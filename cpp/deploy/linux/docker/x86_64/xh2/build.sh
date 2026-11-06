#!/bin/bash

mkdir -p tmp/config
mkdir -p tmp/lib
mkdir -p tmp/bin
cp -r ../../../../../cmake-build-release/bin/libhlie-*.so ./tmp/lib/
cp -r ../../../../../cmake-build-release/bin/houmo-llm-hi-cpp-linux ./tmp/bin/HLIECpp
cp -r ../../../../../config/linux/sys.cfg ./tmp/config/sys.cfg
cp -r ../../../../../config/model_presets.json ./tmp/config/model_presets.json
cp -r ../../../../../config/model_overrides.json ./tmp/config/model_overrides.json
cp -r ../../../../../config/config_schema.json ./tmp/config/config_schema.json
cp -r ../../../../../third_party/pdfium/lib/*.so ./tmp/lib/ 2>/dev/null || true
cp -r ../../../../../third_party/pdfium/bin/icudtl.dat ./tmp/bin/ 2>/dev/null || true

for config_file in sys.cfg model_presets.json model_overrides.json config_schema.json; do
    test -f "./tmp/config/$config_file" || {
        echo "Missing required runtime config: $config_file"
        exit 1
    }
done

DOCKER_REGISTRY="${DOCKER_REGISTRY:-registry.example.invalid}"
docker build -t "${DOCKER_REGISTRY}/application_software/hlie.cpp-xh2-20250910:v0.3.0-x86_64" .

rm -rf tmp
