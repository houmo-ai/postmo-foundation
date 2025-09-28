#!/bin/bash

# 检查参数数量
if [ $# -ne 1 ]; then
    echo "Usage: $0 <xh1|xh2>"
    exit 1
fi
PLATFORM=$1
ARCH=x86_64

# SDK URLs
xh1_dadao_sdk_url="http://10.10.1.53:8082/artifactory/houmo_software/release_v2.4.0/2025/08/14/0200/windows11/houmo_tcim_runtime_xh1_windows11_x86_64-2.4.2.tar.gz"
xh2_dadao_sdk_url="http://10.10.1.53:8082/artifactory/houmo_software/release_xh2_v0.3.0/2025/08/18/1047/windows11/houmo_tcim_runtime_xh2_windows11_x86_64-0.3.0.tar.gz"
xh1_sw_sdk_url="http://10.10.1.53:8082/artifactory/houmo_software/release_v2.4.0/2025/08/14/0200/windows11/houmo_drv_xh1_v2.4.2_windows11_x86_64.zip"
xh2_sw_sdk_url="http://10.10.1.53:8082/artifactory/houmo_software/release_xh2_v0.3.0/2025/08/18/1047/windows11/houmo_drv_xh2_v0.3.0_windows11_x86_64.zip"

xh1_llama_sdk_url="http://10.10.1.53:8082/artifactory/application_software/2025/09/28/1609/windows11/houmo-HLIELLama-xh1_1.2.0_windows11_x86_64.tar.gz"
xh2_llama_sdk_url="http://10.10.1.53:8082/artifactory/application_software/2025/09/28/1609/windows11/houmo-HLIELLama-xh2_1.2.0_windows11_x86_64.tar.gz"

xh2_sd_cpp_sdk_url="http://10.10.1.53:8082/artifactory/application_software/2025/09/28/1609/windows11/houmo-HLIESDCpp-xh2_1.0.0_windows11_x86_64.tar.gz"

# libtorch 下载地址
libtorch_url="http://10.10.1.53:8082/artifactory/thirdparty_local/libtorch/torch.zip"

# 下载文件并检查是否成功
download_file() {
    local url=$1
    local output=$2
    echo "Downloading $url..."
    wget "$url" -O "$output" || { echo "Failed to download $url"; return 1; }
    echo "Download completed: $output"
}

deploy_torch() {
    # 检查torch.zip是否存在
    if [ ! -f "third_party/torch.zip" ]; then
        echo "未找到torch.zip，开始下载..."
        download_file "$libtorch_url" "third_party/torch.zip"
    else
        echo "找到torch.zip文件，跳过下载"
    fi

    # 检查目标目录是否存在，若存在则删除
    if [ -d "third_party/libtorch" ]; then
        echo "删除已存在的 third_party/libtorch 目录..."
        rm -rf third_party/libtorch
    fi

    # 解压文件
    unzip -o third_party/torch.zip -d third_party/
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

    if [ $PLATFORM == "xh2" ]; then
        cp third_party/houmo_drv_xh2/hal/lib/*.dll third_party/houmo/lib/
        mkdir -p third_party/houmo/lib/bin/
        cp third_party/houmo_drv_xh2/hal/lib/*_hal.dll third_party/houmo/lib/bin/cygxh2a_hal.dll

        rm -rf third_party/sw_sdk.zip third_party/houmo_drv*
        # FIXME: 临时解决方案，等待SystemWare SDK更新
        download_file "$xh1_sw_sdk_url" "third_party/sw_sdk.zip"
        unzip third_party/sw_sdk.zip -d third_party/
    fi

    # 移动关键库文件
    if [ -f third_party/houmo_drv_xh1/hal/lib/*hdplrt_asic.dll ]; then
        cp third_party/houmo_drv_xh1/hal/lib/*hdplrt_asic.dll third_party/houmo/lib/
    else
        echo "Warning: libhdplrt_asic.dll not found in expected location"
    fi

    if [ -f third_party/houmo_drv_xh1/hal/lib/*hm800_hal.dll ]; then
        cp third_party/houmo_drv_xh1/hal/lib/*hm800_hal.dll third_party/houmo/lib/
    else
        echo "Warning: libhm800_hal.dll not found in expected location"
    fi

    cp third_party/houmo_drv_xh1/hal/lib/libwinpthread-1.dll third_party/houmo/lib/

    # 清理临时文件
    rm -rf third_party/sw_sdk.zip third_party/houmo_drv*
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

    deploy_torch
    echo "Deployed Dadao SDK and System SDK for $PLATFORM/$ARCH."
}
# 执行主函数
main
