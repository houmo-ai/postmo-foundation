#!/bin/bash
set -e

APP_NAME="HLIECpp"
DEFAULT_INSTALL_DIR="/opt/${APP_NAME}"

# --------------------------
# 检测是否 root 用户
# --------------------------
if [[ $EUID -ne 0 ]]; then
    echo "错误：卸载服务需要 root 权限！"
    echo "请使用 sudo 运行，例如："
    echo "  sudo ./uninstall.sh"
    exit 1
fi

SERVICE_FILE="/etc/systemd/system/${APP_NAME}.service"
INSTALL_DIR=""

# --------------------------
# 停止并删除 systemd 服务
# --------------------------
if [[ -f "$SERVICE_FILE" ]]; then
    echo "停止服务..."
    systemctl stop "${APP_NAME}" || true
    systemctl disable "${APP_NAME}" || true

    # 读取 systemd 服务文件的 WorkingDirectory
    WORKDIR=$(grep '^WorkingDirectory=' "$SERVICE_FILE" | cut -d= -f2)
    if [[ -n "$WORKDIR" ]]; then
        INSTALL_DIR="$WORKDIR"
        echo "检测到安装目录：$INSTALL_DIR"
    fi

    echo "删除 systemd 服务文件..."
    rm -f "$SERVICE_FILE"
    systemctl daemon-reload
else
    echo "警告：${SERVICE_FILE} 不存在，可能未安装或已被删除。"
fi

# --------------------------
# 删除安装目录
# --------------------------
if [[ -z "$INSTALL_DIR" ]]; then
    # 回退到默认路径
    INSTALL_DIR="$DEFAULT_INSTALL_DIR"
    echo "使用默认安装目录：$INSTALL_DIR"
fi

if [[ -d "$INSTALL_DIR" ]]; then
    echo "删除安装目录：$INSTALL_DIR"
    rm -rf "$INSTALL_DIR"
else
    echo "安装目录不存在，跳过删除。"
fi

echo "卸载完成！"