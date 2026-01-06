#!/bin/bash

set -e  # 遇到错误立即退出
set -u  # 使用未定义变量时退出
set -o pipefail  # 管道命令中任何一个失败都算失败

# 检查参数数量
if [ $# -ne 2 ]; then
    echo "Usage: $0 <xh1|xh2> <aarch64|x86_64>"
    exit 1
fi

# 平台和架构参数
PLATFORM=$1
ARCH=$2

# 验证架构参数
case "$ARCH" in
    aarch64|x86_64)
        ;;
    *)
        echo "Invalid architecture: $ARCH"
        exit 1
        ;;
esac

# 验证平台参数
case "$PLATFORM" in
    xh1|xh2)
        ;;
    *)
        echo "Invalid platform: $PLATFORM"
        exit 1
        ;;
esac

# SDK URLs
xh1_dadao_sdk_url="http://10.10.1.53:8082/artifactory/houmo_software/release_xh1_v2.6.0/2025/12/20/0742/ubuntu1804/houmo_tcim_runtime_xh1_ubuntu1804_x86_64-2.6.0.tar.gz"
xh1_dadao_sdk_url_aarch64="http://10.10.1.53:8082/artifactory/houmo_software/release_xh1_v2.6.0/2025/12/20/0742/ubuntu2004/houmo_tcim_runtime_xh1_ubuntu2004_aarch64-2.6.0.tar.gz"
xh2_dadao_sdk_url="http://10.10.1.53:8082/artifactory/houmo_software/release_xh2_v0.7.0/2026/01/04/1914/ubuntu2004/houmo_tcim_runtime_xh2_linux_${ARCH}-0.7.0.tar.gz"

xh2_sw_sdk_url="http://10.10.1.53:8082/artifactory/houmo_software/release_xh2_v0.7.0/2026/01/04/1914/ubuntu1804/houmo-drv-xh2_v0.7.0_linux_x86_64.run"
xh2_sw_sdk_url_aarch64="http://10.10.1.53:8082/artifactory/houmo_software/release_xh2_v0.7.0/2026/01/04/1914/ubuntu2004/houmo-drv-xh2_v0.7.0_linux_aarch64.run"

xh1_llama_sdk_url="http://10.10.1.53:8082/artifactory/application_software/release/llama.cpp/houmo-HLIELLama-xh1_1.3.2_linux_${ARCH}.tar.gz"
xh2_llama_sdk_url="http://10.10.1.53:8082/artifactory/application_software/release/llama.cpp/houmo-HLIELLama-xh2_1.3.2_linux_${ARCH}.tar.gz"

xh2_sd_cpp_sdk_url="http://10.10.1.53:8082/artifactory/application_software/release/sd.cpp/houmo-HLIESDCpp-xh2_1.1.0_linux_${ARCH}.tar.gz"

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

    if [[ $PLATFORM == "xh2" ]]; then
        # 提取SDK
        local sw_url=$2
        download_file "$sw_url" "third_party/sw_sdk.run"
        bash third_party/sw_sdk.run extract --path third_party/

        cp third_party/houmo-sdk/hal/lib/*.so* third_party/houmo/lib/
        cp third_party/houmo-sdk/hal/export/include/*.h third_party/houmo/include/
        rm -rf third_party/houmo-sdk third_party/sw_sdk.run third_party/houmo_drv*
    fi

    # 清理临时文件
    rm -rf third_party/houmo-sdk third_party/sw_sdk.run third_party/houmo_drv*
}


# 主函数
main() {
    # 确定使用的SDK URL
    case "$PLATFORM" in
        xh1)
            deploy_dadao_sdk "$xh1_dadao_sdk_url"
            deploy_llama_sdk "$xh1_llama_sdk_url"
            ;;
        xh2)
            if [ "$ARCH" == "aarch64" ]; then
                deploy_dadao_sdk "$xh2_dadao_sdk_url" "$xh2_sw_sdk_url_aarch64"
            else
                deploy_dadao_sdk "$xh2_dadao_sdk_url" "$xh2_sw_sdk_url"
            fi
            deploy_llama_sdk "$xh2_llama_sdk_url"
            deploy_sd_cpp_sdk "$xh2_sd_cpp_sdk_url"
            ;;
    esac

    echo "Deployed Dadao SDK and System SDK for $PLATFORM/$ARCH."
}
# 执行主函数
main
