#!/bin/bash

# 确保脚本以 root 权限运行
if [ "$(id -u)" -ne 0 ]; then
    echo "请使用 sudo 运行此脚本"
    exit 1
fi

ARCH=$1

# 定义变量
PACKAGE_NAME="HLIECpp-1.4.0-$(date +%Y%m%d%H%M%S)_ubuntu2004_${ARCH}.run"
BUILD_DIR="$(pwd)/build"
SOURCE_DIR="$(pwd)"

chmod +x "$SOURCE_DIR/scripts/preinstall.sh"
chmod +x "$SOURCE_DIR/scripts/postinstall.sh"
chmod +x "$SOURCE_DIR/scripts/uninstall.sh"

# 创建临时构建目录
mkdir -p "$BUILD_DIR"/{bin,lib,etc,demo,config,licenses}
cp "$SOURCE_DIR"/bin/HLIECpp "$BUILD_DIR"/bin/
cp "$SOURCE_DIR"/lib/* "$BUILD_DIR"/lib/ 2>/dev/null || true
cp "$SOURCE_DIR"/etc/* "$BUILD_DIR"/etc/ 2>/dev/null || true
cp "$SOURCE_DIR"/HLIECpp.service "$BUILD_DIR"/
cp "$SOURCE_DIR"/scripts/* "$BUILD_DIR"/
cp "$SOURCE_DIR"/config/* "$BUILD_DIR"/config/
cp -r "$SOURCE_DIR"/licenses/. "$BUILD_DIR"/licenses/

# 创建改进的安装脚本
cat > "$BUILD_DIR/install.sh" <<'EOF'
#!/bin/bash

# 确保以 root 权限运行
if [ "$(id -u)" -ne 0 ]; then
    echo "请使用 sudo 运行此安装程序"
    exit 1
fi

TMP_DIR="/tmp/HLIECpp-$(date +%s)"
mkdir -p "$TMP_DIR"

# 使用更可靠的方式提取文件
ARCHIVE=$(awk '/^__ARCHIVE_BELOW__/ {print NR + 1; exit 0; }' "$0")
tail -n+$ARCHIVE "$0" | gzip -cd > "$TMP_DIR/files.tar" 2>/dev/null || {
    echo "解压失败！" >&2
    exit 1
}

# 解压文件
if ! tar -xf "$TMP_DIR/files.tar" -C "$TMP_DIR"; then
    echo "解压失败！" >&2
    exit 1
fi

# 执行预安装脚本
"$TMP_DIR/preinstall.sh"

mkdir -p /opt/HLIECpp/logs

# 复制文件到目标位置
cp -r "$TMP_DIR/bin" "$TMP_DIR/lib" "$TMP_DIR/etc" "$TMP_DIR/config" "$TMP_DIR/licenses" \
    "$TMP_DIR/HLIECpp.service" "$TMP_DIR/postinstall.sh" /opt/HLIECpp/

# 执行安装后脚本
"$TMP_DIR/postinstall.sh"

# 清理
rm -rf "$TMP_DIR"
exit 0
EOF

# 添加文件分隔标记
echo "__ARCHIVE_BELOW__" >> "$BUILD_DIR/install.sh"

# 打包文件并追加到脚本
cd "$BUILD_DIR" || exit
tar -cf files.tar bin lib etc demo config licenses HLIECpp.service preinstall.sh postinstall.sh
gzip files.tar
cat files.tar.gz >> "$BUILD_DIR/install.sh"
mv "$BUILD_DIR/install.sh" "$SOURCE_DIR/$PACKAGE_NAME"
chmod +x "$SOURCE_DIR/$PACKAGE_NAME"

# 清理构建目录
cd "$SOURCE_DIR" || exit
rm -rf "$BUILD_DIR"

echo "包已创建: $PACKAGE_NAME"
