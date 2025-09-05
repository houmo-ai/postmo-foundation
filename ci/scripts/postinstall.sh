#!/bin/bash

# 复制服务文件并启用服务
sudo cp /opt/HLIECpp/HLIECpp.service /etc/systemd/system/
sudo chmod 777 /etc/systemd/system/HLIECpp.service


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
        rm -rf  /opt/HLIECpp/lib/ld-linux-x86-64.so.2
        rm -rf  /opt/HLIECpp/lib/libc.so.6   /opt/HLIECpp/lib/libm.so.6  /opt/HLIECpp/lib/librt.so.1 /opt/HLIECpp/lib/libdl.so.2  /opt/HLIECpp/lib/libpthread.so.0
        # 检查文件是否存在
        FILE_PATH1="/opt/HLIECpp/lib/ld-linux-x86-64.so.2"  # 需要检查的文件路径
        if [ -e "$FILE_PATH1" ]; then
            echo "错误：文件 $FILE_PATH1 依然已存在！请确认是否使用sudo权限" >&2
            exit 1
        fi
        FILE_PATH2="/opt/HLIECpp/lib/libc.so.6"  # 需要检查的文件路径
        if [ -e "$FILE_PATH2" ]; then
            echo "错误：文件 $FILE_PATH2 依然已存在！请确认是否使用sudo权限" >&2
            exit 1
        fi
        FILE_PATH3="/opt/HLIECpp/lib/libm.so.6"  # 需要检查的文件路径
        if [ -e "$FILE_PATH3" ]; then
            echo "错误：文件 $FILE_PATH3 依然已存在！请确认是否使用sudo权限" >&2
            exit 1
        fi
        FILE_PATH4="/opt/HLIECpp/lib/librt.so.1"  # 需要检查的文件路径
        if [ -e "$FILE_PATH4" ]; then
            echo "错误：文件 $FILE_PATH4 依然已存在！请确认是否使用sudo权限" >&2
            exit 1
        fi
        FILE_PATH5="/opt/HLIECpp/lib/libdl.so.2"  # 需要检查的文件路径
        if [ -e "$FILE_PATH5" ]; then
            echo "错误：文件 $FILE_PATH5 依然已存在！请确认是否使用sudo权限" >&2
            exit 1
        fi

        FILE_PATH6="/opt/HLIECpp/lib/libpthread.so.0"  # 需要检查的文件路径
        if [ -e "$FILE_PATH6" ]; then
            echo "错误：文件 $FILE_PATH6 依然已存在！请确认是否使用sudo权限" >&2
            exit 1
        fi
        echo "++For Ubuntu  delete all unnecessary libs successfully"
        ;;
    "Kylin")
        echo "执行麒麟特定操作..."
        ;;
    *)
        echo "未知系统，当前run包只在麒麟和Ubuntu操作系统测试过，未知系统安装此run包，可能不正常，请知悉"
        #exit 1
        ;;
esac



# 重载 systemd 并启动服务
systemctl daemon-reload
systemctl enable HLIECpp
systemctl start HLIECpp

echo "HoumoInfer 服务已启动！使用以下命令管理："
echo "sudo systemctl start HLIECpp   - 启动服务"
echo "sudo systemctl stop HLIECpp    - 停止服务"
echo "sudo systemctl restart HLIECpp - 重启服务"
echo "sudo systemctl status HLIECpp  - 查看状态"