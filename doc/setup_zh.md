# 安装配置文档

## 系统要求

- **操作系统**：推荐使用 Ubuntu 22.04
- **显卡**：Nvidia 显卡  
- **驱动版本**：建议使用 550 或更高版本  

---

## 1. 创建虚拟环境

建议在虚拟环境中运行训练或部署程序，推荐使用 Conda 创建虚拟环境。如果您的系统中已经安装了 Conda，可以跳过步骤 1.1。

### 1.1 下载并安装 MiniConda

MiniConda 是 Conda 的轻量级发行版，适用于创建和管理虚拟环境。使用以下命令下载并安装：

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```

安装完成后，初始化 Conda：

```bash
~/miniconda3/bin/conda init --all
source ~/.bashrc
```

### 1.2 创建新环境

使用以下命令创建虚拟环境：

```bash
conda create -n unitree_rl_mjlab python=3.11
```

### 1.3 激活虚拟环境

```bash
conda activate unitree_rl_mjlab
```

---

## 2. 安装

### 2.1 下载

通过 Git 克隆仓库：

```bash
git clone https://github.com/unitreerobotics/unitree_rl_mjlab.git
```

### 2.2 安装依赖

```bash
sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev
```

我们将其余所需依赖放入 setup.py 文件中，
进入 unitree_rl_mjlab 项目根目录并安装：

```bash
cd unitree_rl_mjlab
pip install -e .
```

## 总结

### 本地查看推荐的 Stage 5B 离合模型

在仓库根目录运行：

```bash
./tools/remote_fetch_best_wrist_model.sh
./tools/play_best_wrist_model.sh --viewer native --wrist-mode mixed
```

下载脚本当前推荐倍率 12 的候选（尚未通过移动门槛或最终 held-out），
远端来源为 `2026-09-12_20-17-07/model_4999.pt`。本地统一保存到
`models/best_wrist/model.pt`，来源记录在 `model.pt.source.txt`，不再写入
历史 run 目录。替换不同模型时备份旧文件及其来源；原历史文件不会迁移或删除。

运行脚本默认 Python 为 `/mnt/hdd/miniforge3/envs/unitree_rl_mjlab/bin/python`。
若实际环境不同，使用 `G1_MJLAB_PYTHON=/实际路径/bin/python` 指定。
默认配置自动恢复离合、移动范围、肩高和腕部命令，并将课程设到最终难度。
它使用 play 模式的扰动/物理配置，不等同于 held-out evaluator。

`--wrist-mode` 可选 `mixed`（25% 运输、25% 调整、50% 自主平衡），
`transport`、`adjust`、`balance`。单环境随机混合可能连续抽到同一类，
建议分别查看三类。模式保持到回合结束，暂不支持回合内离合切换；
该配置下 GUI 速度滑杆不能覆盖离合命令采样器。
查看旧的无离合模型时同时指定 `G1_WRIST_CHECKPOINT=/旧模型路径`
和 `G1_WRIST_PROFILE=legacy`，不要用新配置加载旧维度检查点。

按照上述步骤完成后，您已经准备好在虚拟环境中运行相关程序。若遇到问题，请参考各组件的官方文档或检查依赖安装是否正确。
