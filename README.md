# Nonlinear2026 Project

Project for window size estimation using linear programming and stochastic gradient descent.

## Project Structure

```
nonlinear2026/
├── data/              # Data files (.npy, .pkl.zip)
├── src/               # Source code modules
│   ├── __init__.py
│   ├── data_loader.py
│   ├── error_calculator.py
│   ├── error_matrix_generator.py
│   ├── H_constructor.py
│   ├── inflection_finder.py
│   ├── LP_solver.py
│   ├── piece_utils.py
│   ├── SGD_solver.py
│   └── y_generator.py
├── scripts/           # Main scripts and entry points
│   ├── main_func.py
│   ├── main_func_test.py
│   └── convert_npy.py
├── results/           # Output files (plots, etc.)
│   └── plots_comparison/
├── tests/             # Test files (to be added)
└── README.md
```

## Usage

### Run main pipeline
```bash
python3 scripts/main_func.py
```

### Run comparison tests
```bash
python3 scripts/main_func_test.py
```

### Convert data files
```bash
python3 scripts/convert_npy.py --in-dir data --out-dir data --fs 700
```

## Requirements

- Python 3.7+
- numpy >= 1.20.0
- scipy >= 1.7.0
- matplotlib >= 3.3.0
- torch >= 1.9.0

安装依赖：
```bash
pip install -r requirements.txt
```

## 在实验室服务器上运行

### 快速开始
查看 [QUICK_START_SERVER.md](QUICK_START_SERVER.md) 获取快速上手指南。

### 详细部署指南
查看 [SERVER_DEPLOYMENT.md](SERVER_DEPLOYMENT.md) 获取完整的服务器部署说明。

### 主要实验脚本

- **训练神经网络模型**: `scripts/train_nn_forward.py`
- **Theta sweep 实验**: `scripts/run_inverse_nn_theta_sweep.py`
- **跨受试者实验**: `scripts/run_cross_subject_theta_sweep.py`

### 批量运行所有实验

```bash
./run_all_experiments.sh
```

或使用 nohup 在后台运行：
```bash
nohup ./run_all_experiments.sh > all_experiments.log 2>&1 &
```

