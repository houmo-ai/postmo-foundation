#!/bin/bash

# 检查参数数量
ARCH=x86_64

# SDK URLs
xh2_dadao_sdk_url="http://10.10.1.53:8082/artifactory/houmo_software/release_xh2_v1.1.0/2026/03/12/2005/windows11/houmo_tcim_runtime_xh2_windows11_x86_64-1.1.0.tar.gz"

xh2_sw_sdk_url="http://10.10.1.53:8082/artifactory/houmo_software/release_xh2_v0.7.0/2025/12/18/1632/windows11/houmo_drv_xh2_v0.7.0_windows11_x86_64.zip"

xh2_llama_sdk_url="http://10.10.1.53:8082/artifactory/application_software/2026/04/03/1804/windows11/houmo-HLIELLama-xh2_2.0.1_windows11_x86_64.tar.gz"

xh2_sd_cpp_sdk_url="http://10.10.1.53:8082/artifactory/application_software/release/sd.cpp/houmo-HLIESDCpp-xh2_1.2.0_windows11_x86_64.tar.gz"
xh2_whisper_cpp_sdk_url="http://10.10.1.53:8082/artifactory/application_software/release/hliewhisper.cpp/houmo-HLIEWhisperCpp-xh2_1.0.0_windows11_x86_64.tar.gz"

uwebsockets_url="http://10.10.1.53:8082/artifactory/application_software/release/hi.cpp/thirdparty/uwebsockets-win_x64.tar.gz"

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

# 部署 Dadao SDK
deploy_dadao_sdk() {
    local dadao_url=$1
    local sw_url=$2

    # 创建或清理第三方库目录
    rm -rf third_party/houmo*
    mkdir -p third_party/houmo

    # 下载并解压 Dadao SDK
    download_file "$dadao_url" "dadao_sdk.tar.gz"
    tar -xzf dadao_sdk.tar.gz -C third_party/houmo --strip-components=1
    cp third_party/houmo/bin/*.dll third_party/houmo/lib/
    rm dadao_sdk.tar.gz

    # 下载并解压 SystemWare SDK
    download_file "$sw_url" "third_party/sw_sdk.zip"

    # 提取SDK
    unzip -o third_party/sw_sdk.zip -d third_party

    cp third_party/houmo_drv_xh2/hal/lib/*.dll third_party/houmo/lib/
    cp third_party/houmo_drv_xh2/hal/lib/*.dll.a third_party/houmo/lib/
    cp third_party/houmo_drv_xh2/hal/export/include/*.h third_party/houmo/include/
    cp third_party/houmo_drv_xh2/hal/include/*.h third_party/houmo/include/
    mkdir -p third_party/houmo/lib/bin/

    rm -rf third_party/sw_sdk.zip third_party/houmo_drv*

    cp third_party/houmo_drv_xh1/hal/lib/libwinpthread-1.dll third_party/houmo/lib/

    # 清理临时文件
    rm -rf third_party/sw_sdk.zip third_party/houmo_drv*
}

# 主函数
main() {
    # 确定使用的SDK URL
    deploy_dadao_sdk "$xh2_dadao_sdk_url" "$xh2_sw_sdk_url"
    deploy_llama_sdk "$xh2_llama_sdk_url"
    deploy_sd_cpp_sdk "$xh2_sd_cpp_sdk_url"
    deploy_whisper_cpp_sdk "$xh2_whisper_cpp_sdk_url"
    deploy_uwebsockets "$uwebsockets_url"

    echo "Deployed Dadao SDK and System SDK for $ARCH."
}
# 执行主函数
main
