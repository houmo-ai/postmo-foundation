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

xh1_llama_sdk_url="http://10.10.1.53:8082/artifactory/application_software/2025/09/02/1725/linux/houmo-application-software-llama.cpp-xh1_1.1.0_windows11_x86_64.tar.gz"
xh2_llama_sdk_url="http://10.10.1.53:8082/artifactory/application_software/2025/09/02/1725/linux/houmo-application-software-llama.cpp-xh2_1.1.0_windows11_x86_64.tar.gz"


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


    # 下载并解压 Dadao SDK
    download_file "$llama_sdk_url" "llama_sdk.tar.gz"
    tar -xzf llama_sdk.tar.gz -C third_party/llama.cpp_houmo --strip-components=1
    rm llama_sdk.tar.gz
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

deploy_other_deps() {
    rm -rf third_party/inih
    mkdir -p third_party/inih/lib
    cp -r /opt/inih/include third_party/inih/
    if [ "$ARCH" == "aarch64" ]; then
        cp /opt/inih/aarch64/* third_party/inih/lib/
    else
        cp /opt/inih/x86_64/* third_party/inih/lib/
    fi

    rm -rf third_party/sd.cpp
    mkdir -p third_party/sd.cpp/lib
    cp -r /opt/sd.cpp/include third_party/sd.cpp/
    if [ "$ARCH" == "aarch64" ]; then
        cp /opt/sd.cpp/aarch64/* third_party/sd.cpp/lib/
    else
        cp /opt/sd.cpp/x86_64/* third_party/sd.cpp/lib/
    fi
}

build_service() {
    echo "Building for ARCH: $ARCH"

    # 创建或清理构建目录
    rm -rf build/*
    if [ $PLATFORM == "xh2" ]; then
        cmake -B build  -DLLAMA_CURL=OFF -DENABLE_HOUMO_XH2=ON
    else
        cmake -B build -DLLAMA_CURL=OFF
    fi
    cmake  --build build --config Release -j 8
    echo "Build completed."
}

package_service() {
    local release_dir="release/$PLATFORM/$ARCH"
    echo "Packaging HLIECPP for $PLATFORM/$ARCH..."

    # 清理并创建目录结构
    rm -rf "$release_dir"/*
    mkdir -p "$release_dir/bin" "$release_dir/config"


    echo "Release files have been organized under: $release_dir"
    # 显示打包内容摘要
    echo "Package contents summary:"
    find "$release_dir" -type f | wc -l
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
            ;;
    esac

    echo "Deployed Dadao SDK and System SDK for $PLATFORM/$ARCH."
    deploy_torch

    # 编译和打包
    build_service
    package_service

    echo "LLAMA build and packaging completed for $PLATFORM/$ARCH."
}
# 执行主函数
main
