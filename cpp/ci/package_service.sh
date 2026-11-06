#!/bin/bash

set -e  # 遇到错误立即退出
set -u  # 使用未定义变量时退出
set -o pipefail  # 管道命令中任何一个失败都算失败

if [ $# -eq 2 ]; then
    ARCH=$2
elif [ $# -eq 1 ]; then
    ARCH=$1
else
    echo "❌ Error: Missing architecture parameter."
    echo "Usage: $0 [to_remove] <aarch64|x86_64>  OR  $0 <aarch64|x86_64>"
    exit 1
fi

ARCH=$(echo "$ARCH" | tr '[:upper:]' '[:lower:]')

# 验证架构参数
case "$ARCH" in
    aarch64|x86_64)
        ;;
    *)
        echo "Invalid architecture: $ARCH"
        exit 1
        ;;
esac

# SDK URLs. Set HOUMO_ARTIFACTORY_BASE_URL in the private build environment.
HOUMO_ARTIFACTORY_BASE_URL="${HOUMO_ARTIFACTORY_BASE_URL:-https://artifacts.example.invalid}"
xh2_dadao_sdk_url="${HOUMO_ARTIFACTORY_BASE_URL}/houmo_software/release_xh2_v1.4.0/2026/07/13/2100/ubuntu2004/houmo_tcim_runtime_xh2_linux_${ARCH}-1.4.0.tar.gz"

xh2_llama_sdk_url="${HOUMO_ARTIFACTORY_BASE_URL}/application_software/release/llama.cpp/houmo-HLIELLama-xh2_2.1.1_linux_${ARCH}.tar.gz"

xh2_sd_cpp_sdk_url="${HOUMO_ARTIFACTORY_BASE_URL}/application_software/release/sd.cpp/houmo-HLIESDCpp-xh2_1.3.0_linux_${ARCH}.tar.gz"
xh2_whisper_cpp_sdk_url="${HOUMO_ARTIFACTORY_BASE_URL}/application_software/release/hliewhisper.cpp/houmo-HLIEWhisperCpp-xh2_1.2.0_linux_${ARCH}.tar.gz"

uwebsockets_url="${HOUMO_ARTIFACTORY_BASE_URL}/application_software/release/hi.cpp/thirdparty/uwebsockets-linux_${ARCH}.tar.gz"
libsamplerate_url="${HOUMO_ARTIFACTORY_BASE_URL}/application_software/release/hi.cpp/thirdparty/libsamplerate-linux_${ARCH}.tar.gz"
eigen3_url="${HOUMO_ARTIFACTORY_BASE_URL}/application_software/release/hi.cpp/thirdparty/eigen3.tar.gz"
kaldi_native_fbank_url="${HOUMO_ARTIFACTORY_BASE_URL}/application_software/release/hi.cpp/thirdparty/kaldi-native-fbank-linux_${ARCH}.tar.gz"
pdfium_url="${HOUMO_ARTIFACTORY_BASE_URL}/application_software/release/hi.cpp/thirdparty/pdfium-linux_${ARCH}.tar.gz"

# 下载文件并检查是否成功
download_file() {
    local url=$1
    local output=$2
    echo "Downloading $url..."
    wget "$url" -O "$output" || { echo "Failed to download $url"; return 1; }
    echo "Download completed: $output"
}

deploy_llama_sdk() {
    local llama_sdk_url=$1
    mkdir -p third_party/llama.cpp_houmo
    rm -rf third_party/llama.cpp_houmo/*


    download_file "$llama_sdk_url" "llama_sdk.tar.gz"
    tar -xzf llama_sdk.tar.gz -C third_party/llama.cpp_houmo --strip-components=1
    rm llama_sdk.tar.gz
}

deploy_sd_cpp_sdk() {
    local sd_cpp_sdk_url=$1
    mkdir -p third_party/sd.cpp
    rm -rf third_party/sd.cpp/*


    download_file "$sd_cpp_sdk_url" "sd.cpp.tar.gz"
    tar -xzf sd.cpp.tar.gz -C third_party/sd.cpp --strip-components=1
    rm sd.cpp.tar.gz
}

deploy_whisper_cpp_sdk() {
    local whisper_cpp_sdk_url=$1
    mkdir -p third_party/whisper.cpp
    rm -rf third_party/whisper.cpp/*


    download_file "$whisper_cpp_sdk_url" "whisper.cpp.tar.gz"
    tar -xzf whisper.cpp.tar.gz -C third_party/whisper.cpp --strip-components=1
    rm whisper.cpp.tar.gz
}

deploy_uwebsockets() {
    local uwebsockets_url=$1
    mkdir -p third_party/uwebsockets
    rm -rf third_party/uwebsockets/*

    download_file "$uwebsockets_url" "uwebsockets.tar.gz"
    tar -xzf uwebsockets.tar.gz -C third_party/uwebsockets --strip-components=1
    rm uwebsockets.tar.gz
}

deploy_libsamplerate() {
    local libsamplerate_url=$1
    mkdir -p third_party/samplerate
    rm -rf third_party/samplerate/*

    download_file "$libsamplerate_url" "libsamplerate.tar.gz"
    tar -xzf libsamplerate.tar.gz -C third_party/samplerate --strip-components=1
    rm libsamplerate.tar.gz
}

deploy_kaldi_native_fbank() {
    local kaldi_native_fbank_url=$1
    mkdir -p third_party/kaldi-native-fbank
    rm -rf third_party/kaldi-native-fbank/*

    download_file "$kaldi_native_fbank_url" "kaldi_native_fbank.tar.gz"
    tar -xzf kaldi_native_fbank.tar.gz -C third_party/kaldi-native-fbank --strip-components=1
    rm kaldi_native_fbank.tar.gz
}

deploy_pdfium() {
    local pdfium_url=$1
    mkdir -p third_party/pdfium
    rm -rf third_party/pdfium/*

    download_file "$pdfium_url" "pdfium.tar.gz"
    tar -xzf pdfium.tar.gz -C third_party/pdfium --strip-components=1
    rm pdfium.tar.gz
}

deploy_eigen3() {
  local eigen3_url=$1
  mkdir -p third_party/eigen3
  rm -rf third_party/eigen3/*

  download_file "$eigen3_url" "eigen3.tar.gz"
  tar -xzf eigen3.tar.gz -C third_party/eigen3 --strip-components=1
  rm eigen3.tar.gz
}

# 部署 Dadao SDK
deploy_dadao_sdk() {
    local dadao_url=$1

    # 创建或清理第三方库目录
    mkdir -p third_party/houmo
    rm -rf third_party/houmo/*

    # 下载并解压 Dadao SDK
    download_file "$dadao_url" "dadao_sdk.tar.gz"
    tar -xzf dadao_sdk.tar.gz -C third_party/houmo --strip-components=1
    rm dadao_sdk.tar.gz
}

deploy_other_deps() {
    rm -rf third_party/inih
    mkdir -p third_party/inih/lib
    cp -r /opt/inih/include third_party/inih/
    if [ "$ARCH" == "aarch64" ]; then
        cp /opt/inih/aarch64/* third_party/inih/lib/
    else
        cp /opt/inih/x86_64/* third_party/inih/lib/
    fi
}

build_service() {
    deploy_other_deps
    echo "Building for ARCH: $ARCH"

    # 创建或清理构建目录
    mkdir -p build
    rm -rf build/*

    cd build
    if [ "$ARCH" == "aarch64" ]; then
        TCIM_RUNTIME_PATH=$(pwd)/../third_party/houmo cmake .. -DCMAKE_TOOLCHAIN_FILE=../ci/toolchain-aarch64.cmake
    else
        TCIM_RUNTIME_PATH=$(pwd)/../third_party/houmo cmake ..
    fi

    if [ $? -ne 0 ]; then
        echo "cmake failed."
        exit 1
    else
        echo "cmake completed."
    fi

    make houmo-llm-hi-cpp-linux -j$(nproc)
    if [ $? -ne 0 ]; then
        echo "make build failed."
        exit 1
    else
        echo "make build completed."
    fi

    cd ..

    echo "Build completed."
}

package_service() {
    local release_dir="release/$ARCH"
    echo "Packaging HLIECPP.service for $ARCH..."

    # 清理并创建目录结构
    rm -rf "$release_dir"/*
    mkdir -p "$release_dir/config" "$release_dir/lib" "$release_dir/bin" "$release_dir/scripts" \
        "$release_dir/samples" "$release_dir/licenses"

    # 复制库文件
    cp build/bin/libhlie*.so "$release_dir/lib/" || true
    cp build/bin/libhlie*.so "$release_dir/lib/" || true
    cp third_party/llama.cpp_houmo/lib/*.so* "$release_dir/lib/" || true
    cp build/bin/houmo-llm-hi-cpp-linux "$release_dir/bin/HLIECpp" || true
    cp samples/zero_shot_prompt.wav "$release_dir/samples/zero_shot_prompt.wav"
    cp config/linux/sys.cfg "$release_dir/config/sys.cfg"
    cp config/model_presets.json "$release_dir/config/model_presets.json"
    cp config/model_overrides.json "$release_dir/config/model_overrides.json"
    cp config/config_schema.json "$release_dir/config/config_schema.json"
    cp config/memory_estimates.json "$release_dir/config/memory_estimates.json"

    local license_manifest="licenses/required-runtime-files.txt"
    if [[ ! -f "$license_manifest" ]]; then
        echo "Error: third-party license manifest is missing: $license_manifest"
        exit 1
    fi
    while IFS= read -r license_file || [[ -n "$license_file" ]]; do
        license_file="${license_file%$'\r'}"
        [[ -z "$license_file" || "$license_file" == \#* ]] && continue
        if [[ ! -f "licenses/$license_file" ]]; then
            echo "Error: required third-party license missing from source tree: $license_file"
            exit 1
        fi
    done < "$license_manifest"
    cp -r licenses/. "$release_dir/licenses/"

    local required_config_files=(
        sys.cfg
        model_presets.json
        model_overrides.json
        config_schema.json
        memory_estimates.json
    )
    for config_file in "${required_config_files[@]}"; do
        if [[ ! -f "$release_dir/config/$config_file" ]]; then
            echo "Error: required runtime config missing from package: $config_file"
            exit 1
        fi
    done

    cp third_party/sd.cpp/lib/libstable-diffusion.so "$release_dir/lib/" || true
    cp third_party/whisper.cpp/lib/libwhisper.so "$release_dir/lib/" || true
    cp third_party/uwebsockets/bin/libuv.so "$release_dir/lib/" || true
    cp third_party/samplerate/lib/libsamplerate.so.* "$release_dir/lib/" || true
    cp third_party/kaldi-native-fbank/lib/libkaldi-native-fbank-core.so "$release_dir/lib/" || true
    local hal_libraries=(third_party/houmo/lib/libhal_xh2a.so*)
    if [[ -e "${hal_libraries[0]}" ]]; then
        cp -a "${hal_libraries[@]}" "$release_dir/lib/"
    else
        echo "Warning: optional HAL runtime library was not found; skipping libhal_xh2a.so*"
    fi
    cp third_party/pdfium/lib/*.so "$release_dir/lib/" || true
    cp third_party/pdfium/bin/icudtl.dat "$release_dir/bin/" || true
    cp third_party/pdfium/bin/icudtl.dat "$release_dir/lib/" || true

    if [[ -f /lib/x86_64-linux-gnu/libstdc++.so.6 && "$ARCH" == "x86_64" ]]; then
        cp /lib/x86_64-linux-gnu/libstdc++.so.6 "$release_dir/lib/"
    fi
    if [[ -f /usr/aarch64-linux-gnu/lib/libstdc++.so.6 && "$ARCH" == "aarch64" ]]; then
        cp /usr/aarch64-linux-gnu/lib/libstdc++.so.6 "$release_dir/lib/"
    fi

    cp ci/create_run_package.sh "$release_dir"
    cp -r ci/scripts "$release_dir"
    cp ci/HLIECpp.service "$release_dir"

    # These files may come from a Windows checkout. Bash heredoc delimiters and
    # shebangs require Unix line endings in the generated Linux package.
    if ! sed -i 's/\r$//' "$release_dir/create_run_package.sh" "$release_dir"/scripts/*.sh; then
        echo "Error: failed to normalize Linux package scripts"
        exit 1
    fi

    if command -v readelf >/dev/null 2>&1; then
        local executable_rpath
        executable_rpath=$(readelf -d "$release_dir/bin/HLIECpp" | grep -E '[(](RPATH|RUNPATH)[)]' || true)
        if [[ "$executable_rpath" != *'$ORIGIN/../lib'* ]]; then
            echo 'Error: packaged HLIECpp is missing the required $ORIGIN/../lib RPATH'
            exit 1
        fi
    fi

    cd "$release_dir"
    if ! bash ./create_run_package.sh "$ARCH"; then
        echo "Error: failed to create Linux run package"
        exit 1
    fi
    cd -

    echo "Release files have been organized under: $release_dir"
    # 显示打包内容摘要
    echo "Package contents summary:"
    find "$release_dir" -type f | wc -l
}

# 主函数
main() {
    # 确定使用的SDK URL
    deploy_dadao_sdk "$xh2_dadao_sdk_url"
    deploy_llama_sdk "$xh2_llama_sdk_url"
    deploy_sd_cpp_sdk "$xh2_sd_cpp_sdk_url"
    deploy_whisper_cpp_sdk "$xh2_whisper_cpp_sdk_url"
    deploy_uwebsockets "$uwebsockets_url"
    deploy_libsamplerate "$libsamplerate_url"
    deploy_eigen3 "$eigen3_url"
    deploy_kaldi_native_fbank "$kaldi_native_fbank_url"
    deploy_pdfium "$pdfium_url"

    echo "Deployed Dadao SDK and System SDK for $ARCH."

    # 编译和打包
    build_service
    package_service

    echo "HLIECpp build and packaging completed for $ARCH."
}
# 执行主函数
main
