# 实验室服务器部署指南

本指南将帮助您在实验室服务器上运行此项目。

## 前置要求

- Python 3.7 或更高版本
- 服务器访问权限（SSH）
- 足够的磁盘空间（用于数据和结果）

## 步骤 1: 上传项目到服务器

### 方法 A: 使用 scp 上传整个项目

```bash
# 在本地终端执行
scp -r /Users/zhiting/Desktop/nonlinear_github username@server_address:/path/to/destination/
```

### 方法 B: 使用 Git（如果项目在 Git 仓库中）

```bash
# 在服务器上执行
cd /path/to/workspace
git clone <your_repo_url>
cd nonlinear_github
```

### 方法 C: 使用 rsync（推荐，支持断点续传）

```bash
# 在本地终端执行
rsync -avz --progress /Users/zhiting/Desktop/nonlinear_github/ username@server_address:/path/to/destination/nonlinear_github/
```

## 步骤 2: 连接到服务器

```bash
ssh username@server_address
cd /path/to/nonlinear_github
```

## 步骤 3: 设置 Python 环境

### 选项 A: 使用 conda（推荐）

```bash
# 创建新的 conda 环境
conda create -n nonlinear python=3.8 -y
conda activate nonlinear

# 安装依赖
pip install -r requirements.txt
```

### 选项 B: 使用 venv

```bash
# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows (如果服务器是 Windows)

# 安装依赖
pip install -r requirements.txt
```

### 选项 C: 使用系统 Python（不推荐，但可用）

```bash
# 直接安装依赖（可能需要 sudo）
pip3 install --user -r requirements.txt
```

## 步骤 4: 验证安装

```bash
python3 -c "import torch; import numpy; import scipy; import matplotlib; print('所有依赖已安装成功！')"
```

## 步骤 5: 检查数据文件

确保数据文件在正确的位置：

```bash
ls -lh data/
# 应该看到 S1.npy, S2.npy, S3.npy 等文件
```

如果数据文件缺失，需要从本地上传：

```bash
# 在本地执行
scp data/*.npy username@server_address:/path/to/nonlinear_github/data/
```

## 步骤 6: 运行项目

### 基本运行

#### 训练神经网络模型

```bash
python3 scripts/train_nn_forward.py
```

#### 运行 theta sweep 实验

```bash
python3 scripts/run_inverse_nn_theta_sweep.py
```

#### 运行跨受试者实验

```bash
python3 scripts/run_cross_subject_theta_sweep.py
```

### 使用 nohup 在后台运行（推荐）

如果实验需要较长时间，建议使用 `nohup` 在后台运行：

```bash
# 运行并保存输出到日志文件
nohup python3 scripts/run_cross_subject_theta_sweep.py > experiment.log 2>&1 &

# 查看进程
ps aux | grep python

# 查看实时日志
tail -f experiment.log
```

### 使用 screen 或 tmux（推荐用于长时间运行）

#### 使用 screen

```bash
# 启动新的 screen 会话
screen -S nonlinear_experiment

# 在 screen 中运行实验
python3 scripts/run_cross_subject_theta_sweep.py

# 按 Ctrl+A 然后按 D 来分离会话

# 重新连接会话
screen -r nonlinear_experiment
```

#### 使用 tmux

```bash
# 启动新的 tmux 会话
tmux new -s nonlinear_experiment

# 在 tmux 中运行实验
python3 scripts/run_cross_subject_theta_sweep.py

# 按 Ctrl+B 然后按 D 来分离会话

# 重新连接会话
tmux attach -t nonlinear_experiment
```

## 步骤 7: 监控运行状态

### 查看 GPU 使用情况（如果使用 GPU）

```bash
# NVIDIA GPU
nvidia-smi

# 持续监控
watch -n 1 nvidia-smi
```

### 查看 CPU 和内存使用情况

```bash
htop
# 或
top
```

### 查看磁盘空间

```bash
df -h
du -sh results/
```

## 步骤 8: 下载结果

实验完成后，将结果下载到本地：

```bash
# 在本地执行
scp -r username@server_address:/path/to/nonlinear_github/results/ ./local_results/
```

或使用 rsync：

```bash
rsync -avz --progress username@server_address:/path/to/nonlinear_github/results/ ./local_results/
```

## 常见问题排查

### 问题 1: 模块导入错误

**错误**: `ModuleNotFoundError: No module named 'torch'`

**解决**:
```bash
# 确保激活了正确的虚拟环境
conda activate nonlinear  # 或 source venv/bin/activate

# 重新安装依赖
pip install -r requirements.txt
```

### 问题 2: CUDA/GPU 相关错误

**错误**: CUDA 不可用或 GPU 相关错误

**解决**:
- 如果服务器没有 GPU，代码会自动使用 CPU
- 如果服务器有 GPU 但未检测到，检查 PyTorch 是否安装了 CUDA 版本：
```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```

### 问题 3: 内存不足

**错误**: `MemoryError` 或 `Killed`

**解决**:
- 减少批处理大小（batch_size）
- 减少迭代次数（num_iters）
- 使用更少的 theta 候选值

### 问题 4: 权限错误

**错误**: 无法写入结果目录

**解决**:
```bash
# 确保有写入权限
chmod -R u+w results/
mkdir -p results/models results/theta_sweep
```

### 问题 5: 数据文件路径错误

**错误**: `FileNotFoundError` 找不到数据文件

**解决**:
```bash
# 检查数据文件是否存在
ls -la data/

# 检查代码中的数据路径设置
grep -r "data/" scripts/
```

## 性能优化建议

1. **使用 GPU**: 如果服务器有 GPU，确保安装了 CUDA 版本的 PyTorch
2. **调整线程数**: 在 CPU 模式下，可以设置环境变量：
   ```bash
   export OMP_NUM_THREADS=4
   export MKL_NUM_THREADS=4
   ```
3. **批量处理**: 如果运行多个实验，考虑使用批处理脚本

## 批处理脚本示例

创建一个 `run_all_experiments.sh` 文件：

```bash
#!/bin/bash

# 激活环境
source venv/bin/activate  # 或 conda activate nonlinear

# 运行实验
echo "开始实验 1: 训练模型"
python3 scripts/train_nn_forward.py

echo "开始实验 2: Theta sweep"
python3 scripts/run_inverse_nn_theta_sweep.py

echo "开始实验 3: 跨受试者实验"
python3 scripts/run_cross_subject_theta_sweep.py

echo "所有实验完成！"
```

然后运行：

```bash
chmod +x run_all_experiments.sh
nohup ./run_all_experiments.sh > all_experiments.log 2>&1 &
```

## 联系支持

如果遇到其他问题，请检查：
1. Python 版本是否符合要求
2. 所有依赖是否正确安装
3. 数据文件是否完整
4. 服务器资源是否充足

