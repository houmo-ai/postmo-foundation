#!/bin/bash

# 检查参数数量
ARCH=x86_64

# SDK URLs
xh2_dadao_sdk_url="http://10.10.1.53:8082/artifactory/houmo_software/release_xh2_v1.4.0/2026/07/13/2100/windows11/houmo_tcim_runtime_xh2_windows11_x86_64-1.4.0.tar.gz"

xh2_llama_sdk_url="http://10.10.1.53:8082/artifactory/application_software/release/llama.cpp/houmo-HLIELLama-xh2_2.1.1_windows11_x86_64.tar.gz"

xh2_sd_cpp_sdk_url="http://10.10.1.53:8082/artifactory/application_software/release/sd.cpp/houmo-HLIESDCpp-xh2_1.3.0_windows11_x86_64.tar.gz"
xh2_whisper_cpp_sdk_url="http://10.10.1.53:8082/artifactory/application_software/release/hliewhisper.cpp/houmo-HLIEWhisperCpp-xh2_1.2.0_windows11_x86_64.tar.gz"

uwebsockets_url="http://10.10.1.53:8082/artifactory/application_software/release/hi.cpp/thirdparty/uwebsockets-win_x64.tar.gz"
libsamplerate_url="http://10.10.1.53:8082/artifactory/application_software/release/hi.cpp/thirdparty/libsamplerate-win_x64.tar.gz"
eigen3_url="http://10.10.1.53:8082/artifactory/application_software/release/hi.cpp/thirdparty/eigen3.tar.gz"
kaldi_native_fbank_url="http://10.10.1.53:8081/artifactory/application_software/release/hi.cpp/thirdparty/kaldi-native-fbank-win_x64.tar.gz"
pdfium_url="http://10.10.1.53:8082/artifactory/application_software/release/hi.cpp/thirdparty/pdfium-win_x64.tar.gz"

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

deploy_eigen3() {
  local eigen3_url=$1
  mkdir -p third_party/eigen3
  rm -rf third_party/eigen3/*

  download_file "$eigen3_url" "eigen3.tar.gz"
  tar -xzf eigen3.tar.gz -C third_party/eigen3 --strip-components=1
  rm eigen3.tar.gz
}

deploy_pdfium() {
    local pdfium_url=$1
    if [ -z "$pdfium_url" ]; then
        echo "PDFIUM_URL is not set, skip deploying PDFium."
        return 0
    fi

    mkdir -p third_party/pdfium
    rm -rf third_party/pdfium/*

    download_file "$pdfium_url" "pdfium.tar.gz"
    tar -xzf pdfium.tar.gz -C third_party/pdfium --strip-components=1
    rm pdfium.tar.gz
}

# 部署 Dadao SDK
deploy_dadao_sdk() {
    local dadao_url=$1

    # 创建或清理第三方库目录
    rm -rf third_party/houmo*
    mkdir -p third_party/houmo

    # 下载并解压 Dadao SDK
    download_file "$dadao_url" "dadao_sdk.tar.gz"
    tar -xzf dadao_sdk.tar.gz -C third_party/houmo --strip-components=1
    cp third_party/houmo/bin/*.dll third_party/houmo/lib/
    rm dadao_sdk.tar.gz
}

build_service() {
    echo "Building for ARCH: $ARCH"

    # 创建或清理构建目录
    rm -rf build/*
    TCIM_RUNTIME_PATH=$(pwd)/third_party/houmo cmake -B build

    if [ $? -ne 0 ]; then
        echo "cmake failed."
        exit 1
    else
        echo "cmake completed."
    fi

    cmake  --build build --target houmo-llm-hi-cpp-win --config Release -j 8
    if [ $? -ne 0 ]; then
        echo "make build failed."
        exit 1
    else
        echo "make build completed."
    fi

    echo "Build completed."
}

package_service() {
    local release_dir="release/$ARCH"
    echo "Packaging HLIECPP for $ARCH..."

    # 清理并创建目录结构
    rm -rf "$release_dir"/*
    mkdir -p "$release_dir/bin" "$release_dir/config" "$release_dir/samples" "$release_dir/licenses"
    cp build/bin/Release/houmo-llm-hi-cpp-win.exe "$release_dir/bin/HLIECpp.exe"
    cp samples/zero_shot_prompt.wav "$release_dir/samples/zero_shot_prompt.wav"
    cp build/bin/Release/hlie-*.dll "$release_dir/bin/"
    cp third_party/llama.cpp_houmo/bin/*.dll "$release_dir/bin/"
    cp config/win/sys.cfg "$release_dir/config/"
    cp config/model_presets.json "$release_dir/config/"
    cp config/model_overrides.json "$release_dir/config/"
    cp config/config_schema.json "$release_dir/config/"
    cp config/memory_estimates.json "$release_dir/config/"

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

    cp third_party/sd.cpp/bin/stable-diffusion.dll "$release_dir/bin/"
    cp third_party/whisper.cpp/bin/whisper.dll "$release_dir/bin/"
    cp third_party/uwebsockets/bin/uv.dll "$release_dir/bin/"
    cp third_party/samplerate/bin/samplerate.dll "$release_dir/bin/"
    cp third_party/kaldi-native-fbank/bin/kaldi-native-fbank-core.dll "$release_dir/bin/" || true
    cp third_party/pdfium/bin/*.dll "$release_dir/bin/" || true
    cp third_party/pdfium/bin/icudtl.dat "$release_dir/bin/" || true

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

    echo "LLAMA build and packaging completed for $ARCH."
}
# 执行主函数
main
