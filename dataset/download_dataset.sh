#!/bin/bash

# 数据集下载脚本
# 用法: ./download_dataset.sh

DATASET_URL="https://cuhko365-my.sharepoint.com/:u:/g/personal/122090714_link_cuhk_edu_cn/EVHRXfzXJV5MmYXo5VkXiX8BdsKhGmE5C2eRUCdfO68kew?e=2QF8cn"
DATASET_ZIP="dataset.zip"
TARGET_DIR="dataset"

echo "=== 正在下载数据集... ==="
echo "来源: $DATASET_URL"
echo "目标: ./$TARGET_DIR/"

# 创建目标目录
mkdir -p "$TARGET_DIR"

# 使用 wget 下载
if ! wget --no-check-certificate "$DATASET_URL" -O "$DATASET_ZIP"; then
    echo "❌ 下载失败！请检查链接是否有效，或尝试手动下载。"
    exit 1
fi

# 解压到 dataset/
echo "=== 正在解压... ==="
if ! unzip -q "$DATASET_ZIP" -d "$TARGET_DIR"; then
    echo "❌ 解压失败！请检查 ZIP 文件是否损坏。"
    exit 1
fi

# 清理临时文件
rm "$DATASET_ZIP"

echo "✅ 数据集下载并解压完成！路径: ./$TARGET_DIR/"
