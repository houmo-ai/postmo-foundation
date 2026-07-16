#!/bin/bash

# ==============================
# 安装脚本
# 可选参数：
#   ./install.sh [INSTALL_DIR]
#   默认 INSTALL_DIR=/opt
# ==============================

INSTALL_DIR=${1:-/opt}  # 若用户未传参数，则使用 /opt 作为默认安装路径
SERVICE_NAME="HLIECpp"

echo "安装路径: ${INSTALL_DIR}"

# 服务文件路径
SRC_SERVICE_FILE="${INSTALL_DIR}/${SERVICE_NAME}/${SERVICE_NAME}.service"
DST_SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

# 检查服务文件是否存在
if [ ! -f "$SRC_SERVICE_FILE" ]; then
    echo "错误: 未找到服务文件 $SRC_SERVICE_FILE"
    exit 1
fi

# 如果安装路径不是 /opt，则修改服务文件中的路径
if [ "$INSTALL_DIR" != "/opt" ]; then
    echo "检测到自定义安装路径，正在替换 .service 文件中的路径..."
    # 临时文件用于替换
    TMP_SERVICE_FILE=$(mktemp)

    # 用 sed 将 /opt 替换为新的 INSTALL_DIR
    sudo sed "s#/opt#${INSTALL_DIR}#g" "$SRC_SERVICE_FILE" > "$TMP_SERVICE_FILE"

    # 复制修改后的文件到系统服务目录
    sudo cp "$TMP_SERVICE_FILE" "$DST_SERVICE_FILE"

    # 删除临时文件
    rm -f "$TMP_SERVICE_FILE"
else
    # 默认路径，直接复制
    sudo cp "$SRC_SERVICE_FILE" "$DST_SERVICE_FILE"
fi

sudo chmod 777 /etc/systemd/system/${SERVICE_NAME}.service
sudo chmod 777 -R ${INSTALL_DIR}/${SERVICE_NAME}/config
sudo mkdir -p /opt/models
sudo chmod 777 -R /opt/models

check_os() {
    # 方法1: 检查/etc/os-release
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        if [[ "$ID" == "ubuntu" ]]; then
            echo "Ubuntu"
            return 0
        elif [[ "$ID" == "kylin" || "$ID_LIKE" == *"kylin"* ]]; then
            echo "Kylin"
            return 0
        fi
    fi

    # 方法2: 检查/etc/lsb-release
    if [ -f /etc/lsb-release ]; then
        . /etc/lsb-release
        if [[ "$DISTRIB_ID" == "Ubuntu" ]]; then
            echo "Ubuntu"
            return 0
        elif [[ "$DISTRIB_ID" == "Kylin" ]]; then
            echo "Kylin"
            return 0
        fi
    fi

    # 方法3: 检查特定文件
    if [ -f /etc/kylin-release ]; then
        echo "Kylin"
        return 0
    fi

    echo "Unknown"
    return 1
}

echo "calling os check"
os_type=$(check_os)
echo "检测结果: $os_type"

# 根据检测结果执行不同操作
case "$os_type" in
    "Ubuntu")
        echo "执行Ubuntu特定操作..."

        rm -rf  "${INSTALL_DIR}/${SERVICE_NAME}/lib/ld-linux-x86-64.so.2"
        rm -rf  "${INSTALL_DIR}/${SERVICE_NAME}/lib/libc.so.6" \
                 "${INSTALL_DIR}/${SERVICE_NAME}/lib/libm.so.6" \
                 "${INSTALL_DIR}/${SERVICE_NAME}/lib/librt.so.1" \
                 "${INSTALL_DIR}/${SERVICE_NAME}/lib/libdl.so.2" \
                 "${INSTALL_DIR}/${SERVICE_NAME}/lib/libpthread.so.0"

        # 检查文件是否存在
        for file in \
            "ld-linux-x86-64.so.2" \
            "libc.so.6" \
            "libm.so.6" \
            "librt.so.1" \
            "libdl.so.2" \
            "libpthread.so.0"; do
            FILE_PATH="${INSTALL_DIR}/${SERVICE_NAME}/lib/${file}"
            if [ -e "$FILE_PATH" ]; then
                echo "错误：文件 $FILE_PATH 依然已存在！请确认是否使用sudo权限" >&2
                exit 1
            fi
        done

        echo "++For Ubuntu delete all unnecessary libs successfully"
        ;;
    "Kylin")
        echo "执行麒麟特定操作..."
        ;;
    *)
        echo "未知系统，当前run包只在麒麟和Ubuntu操作系统测试过，未知系统安装此run包，可能不正常，请知悉"
        ;;
esac

# 重载 systemd 并启动服务
sudo systemctl daemon-reload
sudo systemctl enable ${SERVICE_NAME}
sudo systemctl start ${SERVICE_NAME}

echo "HoumoInfer 服务已启动！使用以下命令管理："
echo "sudo systemctl start ${SERVICE_NAME}   - 启动服务"
echo "sudo systemctl stop ${SERVICE_NAME}    - 停止服务"
echo "sudo systemctl restart ${SERVICE_NAME} - 重启服务"
echo "sudo systemctl status ${SERVICE_NAME}  - 查看状态"
