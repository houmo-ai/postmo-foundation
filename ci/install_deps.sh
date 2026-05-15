#!/bin/bash

set -e  # 遇到错误立即退出
set -u  # 使用未定义变量时退出
set -o pipefail  # 管道命令中任何一个失败都算失败

# 检查参数数量
if [ $# -ne 1 ]; then
    echo "Usage: $0 <aarch64|x86_64>"
    exit 1
fi

# 平台和架构参数
ARCH=$1

# 验证架构参数
case "$ARCH" in
    aarch64|x86_64)
        ;;
    *)
        echo "Invalid architecture: $ARCH"
        exit 1
        ;;
esac

# SDK URLs
xh2_dadao_sdk_url="http://10.10.1.53:8082/artifactory/houmo_software/release_xh2_v1.3.0/2026/05/11/2348/ubuntu2004/houmo_tcim_runtime_xh2_linux_${ARCH}-1.3.0.tar.gz"

xh2_sw_sdk_url="http://10.10.1.53:8082/artifactory/houmo_software/release_xh2_v1.3.0/2026/05/11/2348/ubuntu1804/houmo-drv-xh2_v1.3.0_linux_x86_64.run"
xh2_sw_sdk_url_aarch64="http://10.10.1.53:8082/artifactory/houmo_software/release_xh2_v1.3.0/2026/05/11/2348/ubuntu2004/houmo-drv-xh2_v1.3.0_linux_aarch64.run"

xh2_llama_sdk_url="http://10.10.1.53:8082/artifactory/application_software/2026/05/15/1829/linux/houmo-HLIELLama-xh2_2.1.0_linux_${ARCH}.tar.gz"

xh2_sd_cpp_sdk_url="http://10.10.1.53:8082/artifactory/application_software/release/sd.cpp/houmo-HLIESDCpp-xh2_1.3.0_linux_${ARCH}.tar.gz"
xh2_whisper_cpp_sdk_url="http://10.10.1.53:8082/artifactory/application_software/release/hliewhisper.cpp/houmo-HLIEWhisperCpp-xh2_1.1.0_linux_${ARCH}.tar.gz"

uwebsockets_url="http://10.10.1.53:8082/artifactory/application_software/release/hi.cpp/thirdparty/uwebsockets-linux_${ARCH}.tar.gz"
libsamplerate_url="http://10.10.1.53:8082/artifactory/application_software/release/hi.cpp/thirdparty/libsamplerate-linux_${ARCH}.tar.gz"
eigen3_url="http://10.10.1.53:8082/artifactory/application_software/release/hi.cpp/thirdparty/eigen3.tar.gz"
kaldi_native_fbank_url="http://10.10.1.53:8082/artifactory/application_software/release/hi.cpp/thirdparty/kaldi-native-fbank-linux_${ARCH}.tar.gz"


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

deploy_whisper_cpp_sdk() {
    local whisper_cpp_sdk_url=$1
    mkdir -p third_party/whisper.cpp
    rm -rf third_party/whisper.cpp/*


    download_file "$whisper_cpp_sdk_url" "whisper.cpp.tar.gz"
    tar -xzf whisper.cpp.tar.gz -C third_party/whisper.cpp --strip-components=1
    rm whisper.cpp.tar.gz
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

    # 提取SDK
    local sw_url=$2
    download_file "$sw_url" "third_party/sw_sdk.run"
    bash third_party/sw_sdk.run extract --path third_party/

    cp third_party/houmo-sdk/hal/lib/*.so* third_party/houmo/lib/
    cp third_party/houmo-sdk/hal/export/include/*.h third_party/houmo/include/

    # 清理临时文件
    rm -rf third_party/houmo-sdk third_party/sw_sdk.run third_party/houmo_drv* third_party/houmo-drv*
}


# 主函数
main() {
    # 确定使用的SDK URL
    if [ "$ARCH" == "aarch64" ]; then
        deploy_dadao_sdk "$xh2_dadao_sdk_url" "$xh2_sw_sdk_url_aarch64"
    else
        deploy_dadao_sdk "$xh2_dadao_sdk_url" "$xh2_sw_sdk_url"
    fi
    deploy_llama_sdk "$xh2_llama_sdk_url"
    deploy_sd_cpp_sdk "$xh2_sd_cpp_sdk_url"
    deploy_whisper_cpp_sdk "$xh2_whisper_cpp_sdk_url"
    deploy_uwebsockets "$uwebsockets_url"
    deploy_libsamplerate "$libsamplerate_url"
    deploy_eigen3 "$eigen3_url"
    deploy_kaldi_native_fbank "$kaldi_native_fbank_url"

    echo "Deployed Dadao SDK and System SDK for $ARCH."
}
# 执行主函数
main
