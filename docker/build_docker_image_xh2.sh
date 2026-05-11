#!/bin/bash
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # 无色

# ================== 按系统版本+架构定义的下载链接表 ==================
declare -A DOWNLOAD_URLS

# 从download_resource_url.json中读取下载链接并存入关联数组
while IFS=": " read -r key value; do
    # 去除可能的引号和逗号
    key=$(echo "$key" | tr -d '",')
    value=$(echo "$value" | tr -d '",')
    DOWNLOAD_URLS["$key"]="$value"
done < <(grep -E '"driver"|"tcim_runtime"|"hlievllm"|"hlievllm-plugin"|"hliepython"' download_resource_url.json)

# ------------------ 获取镜像版本 ------------------
read -ep "$(echo -e ${YELLOW}请输入要构建的镜像版本（如：v2.5.0）: ${NC})" version
IMAGE_VERSION="${version}"
IMAGE_TAG="harbor.houmo.ai/application_software/infer:${IMAGE_VERSION}"

# ------------------ 获取系统架构 ------------------
ARCH=$(uname -m)
if [[ "$ARCH" != "x86_64" && "$ARCH" != "aarch64" ]]; then
    echo -e "${RED}❌ 不支持的架构: $ARCH${NC}"
    exit 1
fi

# ------------------ 获取系统版本 ------------------
OS_VER="24.04" # 目前我们只构建基于 Ubuntu 24.04 的镜像，后续可以根据需要添加对其他版本的支持

# ------------------ 获取下载链接 ------------------
DRIVER_URL=${DOWNLOAD_URLS["driver"]}
TCIM_RUNTIME_URL=${DOWNLOAD_URLS["tcim_runtime"]}
HLIEVLLM_URL=${DOWNLOAD_URLS["hlievllm"]}
HLIEVLLM_PLUGIN_URL=${DOWNLOAD_URLS["hlievllm-plugin"]}
HLIEPYTHON_URL=${DOWNLOAD_URLS["hliepython"]}

if [ -z "$DRIVER_URL" ] || [ -z "$TCIM_RUNTIME_URL" ] || [ -z "$HLIEVLLM_URL" ] || [ -z "$HLIEVLLM_PLUGIN_URL" ] || [ -z "$HLIEPYTHON_URL" ]; then
    echo -e "${RED}❌ 未找到对应架构(${ARCH}) 和系统(${OS_VER}) 的下载链接${NC}"
    exit 1
fi

# ===== 📦 下载 Houmo toolchain =====
DOWNLOAD_DIR="./houmo"
mkdir -p "$DOWNLOAD_DIR"

download_file() {
    local url=$1
    local filename=$(basename "$url")
    local filepath="${DOWNLOAD_DIR}/${filename}"

    if [ -f "$filepath" ]; then
        echo -e "${GREEN}✅ $filename 已存在于 $DOWNLOAD_DIR，跳过下载${NC}"  # 打印给用户
    else
        echo -e "${YELLOW}📥 正在下载 $filename 到 $DOWNLOAD_DIR...${NC}"
        wget -O "$filepath" "$url" || { echo -e "${RED}❌ 下载 $filename 失败！${NC}"; exit 1; }
        echo -e "${GREEN}✅ 下载完成：$filepath${NC}"
    fi

    # 只输出纯路径给调用者
    echo "$filepath"
}

echo -e "${BLUE}🚀 开始批量下载必要组件...${NC}"
houmo_drv_file=$(download_file "$DRIVER_URL")
houmo_tcim_runtime_file=$(download_file "$TCIM_RUNTIME_URL")
hlievllm_file=$(download_file "$HLIEVLLM_URL")
hlievllm_plugin_file=$(download_file "$HLIEVLLM_PLUGIN_URL")
hliepython_file=$(download_file "$HLIEPYTHON_URL")
echo -e "${GREEN}✅ 所有组件下载完成${NC}\n"

# ------------------ 🐳 Docker Build ------------------
DOCKERFILE_PATH="Dockerfile/Dockerfile.xh2_ubuntu${OS_VER}_${ARCH}"
echo -e "${BLUE}⚙️ 选用 Dockerfile: $DOCKERFILE_PATH${NC}"
echo -e "${YELLOW}🔧 开始构建 Docker 镜像：${IMAGE_TAG}${NC}"
start_time=$(date +%s)

docker build \
  -f ${DOCKERFILE_PATH} \
  --build-arg houmo_drv="$(basename "$houmo_drv_file")" \
  --build-arg houmo_tcim_runtime="$(basename "$houmo_tcim_runtime_file")" \
  --build-arg hlievllm="$(basename "$hlievllm_file")" \
  --build-arg hlievllm_plugin="$(basename "$hlievllm_plugin_file")" \
  --build-arg hliepython="$(basename "$hliepython_file")" \
  -t "${IMAGE_TAG}" \
  .

end_time=$(date +%s)
build_duration=$((end_time - start_time))

echo -e "${GREEN}✅ Docker 镜像构建完成：${IMAGE_TAG}${NC}"
echo -e "${BLUE}⏱️ 用时：${build_duration} 秒${NC}"
