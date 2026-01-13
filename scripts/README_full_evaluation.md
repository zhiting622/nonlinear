# 完整评估脚本使用说明

## 概述

`run_full_evaluation.py` 实现了完整的实验评估计划，包括：

- **Phase 0**: 实验协议配置（已锁定）
- **Phase 1.1**: 3折交叉验证（主要评估）
- **Phase 1.2**: 优化稳定性测试（有限的蒙特卡洛）- 待实现
- **Phase 1.3**: 数据消融测试（可选）- 待实现

## 使用方法

### 基本运行

```bash
cd /isis/home/zhouz25/projects/nonlinear
python scripts/run_full_evaluation.py
```

### 配置说明

实验配置保存在 `experimental_protocol.json` 中，包括：

- **Theta候选值**: 4-30的整数
- **Ground truth thetas**: 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30
- **随机种子**: 42（确保可重现性）
- **优化器**: Adam, lr=0.01, 1000 iterations
- **正则化**: TV regularization, lam=0.01

### 输出结构

结果保存在 `results/full_evaluation/phase_1_1_cross_validation/` 目录下：

```
phase_1_1_cross_validation/
├── folds_info.json                    # 折信息（训练/测试受试者分配）
├── aggregate_statistics.json          # 聚合统计信息
├── summary_table.json                 # 汇总表（每个评估的theta_hat和error）
├── all_results.json                    # 完整结果
├── plots/                              # 可视化图表
│   ├── error_distribution.png         # 错误分布直方图
│   ├── error_by_subject_boxplot.png   # 按受试者的错误箱线图
│   └── example_error_curves.png        # 示例错误曲线（好/中/差）
└── fold_1/                             # 第1折的结果
    ├── fold_results.json
    └── intermediate_*.json            # 每个theta候选的中间结果
```

### 关键特性

1. **完全可重现**: 使用固定随机种子（42）
2. **详细输出**: 显示每个步骤的进度和估计时间
3. **中间结果保存**: 每个theta候选完成后立即保存结果
4. **Knee检测**: 使用斜率过零点方法检测knee点
5. **完整记录**: 记录所有错误曲线、斜率和knee检测信息

### 实验设计

- **3折交叉验证**: 每折训练5个受试者，测试10个受试者
- **15个受试者**: S1-S15
- **每个测试受试者**: 对每个ground truth theta运行完整的theta sweep
- **Knee检测**: 计算连续点之间的斜率，找到斜率过零点

### 性能优化

- 使用所有56个CPU核心（通过PyTorch的线程设置）
- 每个theta候选完成后立即保存，可以跟踪进度
- 详细的时间估计和进度显示

## 注意事项

1. 确保所有15个受试者的数据文件（S1.npy - S15.npy）都在 `data/` 目录下
2. 实验可能需要较长时间（取决于数据大小和计算资源）
3. 中间结果会自动保存，即使中断也可以从已完成的部分恢复信息

