#!/bin/bash

# ==============================
# 安装前清理脚本
# 可选参数：
#   ./pre_install.sh [INSTALL_DIR]
# 默认 INSTALL_DIR=/opt
# ==============================

set -e

INSTALL_DIR=${1:-/opt}
SERVICE_NAME="HLIECpp"

SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
INSTALL_PATH="${INSTALL_DIR}/${SERVICE_NAME}"
UNINSTALL_SCRIPT="${INSTALL_PATH}/scripts/uninstall.sh"

echo "===== Pre-install 开始 ====="
echo "安装路径: ${INSTALL_DIR}"

# 必须 root 运行
if [ "$EUID" -ne 0 ]; then
    echo "请使用 sudo 运行此脚本"
    exit 1
fi


# ️调用旧版本 uninstall.sh（如果存在）
if [ -f "$UNINSTALL_SCRIPT" ]; then
    echo "检测到旧版本 uninstall.sh，执行卸载..."
    chmod +x "$UNINSTALL_SCRIPT"
    "$UNINSTALL_SCRIPT" || true
else
    echo "未发现旧版本 uninstall.sh，跳过"
fi

# 杀掉残留进程（防止僵尸）
if pgrep -f "${SERVICE_NAME}" > /dev/null; then
    echo "发现残留进程，正在强制清理..."
    pkill -9 -f "${SERVICE_NAME}" || true
fi

# 重新加载 systemd
systemctl daemon-reload

echo "===== Pre-install 清理完成 ====="