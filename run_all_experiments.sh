#!/bin/bash

# 批量运行所有实验的脚本
# 使用方法: ./run_all_experiments.sh

set -e  # 遇到错误立即退出

echo "=========================================="
echo "开始运行所有实验"
echo "=========================================="
echo ""

# 检查 Python 环境
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 python3"
    exit 1
fi

# 检查数据文件
if [ ! -d "data" ] || [ -z "$(ls -A data/*.npy 2>/dev/null)" ]; then
    echo "警告: 数据文件可能缺失，请检查 data/ 目录"
fi

# 创建结果目录
mkdir -p results/models
mkdir -p results/theta_sweep
mkdir -p results/plots_comparison

echo "步骤 1: 训练神经网络模型..."
python3 scripts/train_nn_forward.py
echo "✓ 模型训练完成"
echo ""

echo "步骤 2: 运行 theta sweep 实验..."
python3 scripts/run_inverse_nn_theta_sweep.py
echo "✓ Theta sweep 完成"
echo ""

echo "步骤 3: 运行跨受试者实验..."
python3 scripts/run_cross_subject_theta_sweep.py
echo "✓ 跨受试者实验完成"
echo ""

echo "=========================================="
echo "所有实验完成！"
echo "=========================================="
echo ""
echo "结果保存在 results/ 目录中："
echo "  - 模型: results/models/"
echo "  - Theta sweep 结果: results/theta_sweep/"
echo "  - 对比图: results/plots_comparison/"
echo ""

