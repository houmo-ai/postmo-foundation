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
xh1_dadao_sdk_url="http://10.10.1.53:8082/artifactory/houmo_software/release_v2.4.0/2025/08/14/0200/ubuntu2004/houmo_tcim_runtime_xh1_ubuntu2004_${ARCH}-2.4.2.tar.gz"
xh2_dadao_sdk_url="http://10.10.1.53:8082/artifactory/houmo_software/release_xh2_v0.5.0/2025/10/28/1950/ubuntu2004/houmo_tcim_runtime_xh2_ubuntu2004_${ARCH}-0.5.0.tar.gz"
xh1_sw_sdk_url="http://10.10.1.53:8082/artifactory/houmo_software/release_v2.4.0/2025/09/19/0200/ubuntu2004/houmo_drv_xh1_v2.4.3_ubuntu2004_${ARCH}.run"
#FIXME:
xh2_sw_sdk_url="http://10.10.1.53:8082/artifactory/houmo_software/release_xh2_v0.5.0/2025/10/28/1950/ubuntu2004/houmo_drv_xh2_v0.5.0_ubuntu2004_${ARCH}.run"

xh1_llama_sdk_url="http://10.10.1.53:8082/artifactory/application_software/2025/12/03/2330/linux/houmo-HLIELLama-xh1_1.3.0_linux_${ARCH}.tar.gz"
xh2_llama_sdk_url="http://10.10.1.53:8082/artifactory/application_software/2025/12/03/2330/linux/houmo-HLIELLama-xh2_1.3.0_linux_${ARCH}.tar.gz"

xh2_sd_cpp_sdk_url="http://10.10.1.53:8082/artifactory/application_software/2025/12/03/2330/linux/houmo-HLIESDCpp-xh2_1.1.0_linux_${ARCH}.tar.gz"

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
    local sw_url=$2

    # 创建或清理第三方库目录
    mkdir -p third_party/houmo
    rm -rf third_party/houmo/*

    # 下载并解压 Dadao SDK
    download_file "$dadao_url" "dadao_sdk.tar.gz"
    tar -xzf dadao_sdk.tar.gz -C third_party/houmo --strip-components=1
    rm dadao_sdk.tar.gz


    # 提取SDK
    download_file "$sw_url" "third_party/sw_sdk.run"
    bash third_party/sw_sdk.run extract --path third_party/


    if [[ $PLATFORM == "xh2" ]]; then
        cp third_party/houmo-sdk/hal/lib/*.so* third_party/houmo/lib/
        cp third_party/houmo-sdk/hal/export/*.h third_party/houmo/include/
        rm -rf third_party/houmo-sdk third_party/sw_sdk.run third_party/houmo_drv*
    fi

    if [ $PLATFORM == "xh2" ]; then
        # FIXME: xh2 还需下载xh1的包，临时解决方案，等待SystemWare SDK更新
        download_file "$xh1_sw_sdk_url" "third_party/sw_sdk.run"
        bash third_party/sw_sdk.run extract --path third_party/
    fi

    # 移动关键库文件
    if [ -f third_party/houmo-sdk/hal/lib/libhdplrt_asic.so ]; then
        cp third_party/houmo-sdk/hal/lib/libhdplrt_asic.so third_party/houmo/lib/
    else
        echo "Warning: libhdplrt_asic.so not found in expected location"
    fi

    if [ -f third_party/houmo-sdk/hal/lib/libhm800_hal.so ]; then
        cp third_party/houmo-sdk/hal/lib/libhm800_hal.so third_party/houmo/lib/
    else
        echo "Warning: libhm800_hal.so not found in expected location"
    fi

    # 清理临时文件
    rm -rf third_party/houmo-sdk third_party/sw_sdk.run third_party/houmo_drv*
}


# 主函数
main() {
    # 确定使用的SDK URL
    case "$PLATFORM" in
        xh1)
            deploy_dadao_sdk "$xh1_dadao_sdk_url" "$xh1_sw_sdk_url"
            deploy_llama_sdk "$xh1_llama_sdk_url"
            ;;
        xh2)
            deploy_dadao_sdk "$xh2_dadao_sdk_url" "$xh2_sw_sdk_url"
            deploy_llama_sdk "$xh2_llama_sdk_url"
            deploy_sd_cpp_sdk "$xh2_sd_cpp_sdk_url"
            ;;
    esac

    echo "Deployed Dadao SDK and System SDK for $PLATFORM/$ARCH."
}
# 执行主函数
main
