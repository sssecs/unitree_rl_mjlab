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

### 本地查看推荐低位工作空间模型

当前探索候选为 operation capability pack 的 balanced 模型
（`2026-09-14_11-27-20/model_9998.pt`），保存在
`models/best_ground/model.pt`（附来源记录）。下载默认覆盖旧本地候选，不备份；
远端 checkpoint 保留。播放默认启用完整课程的连续 XYZ／朝向、肩高和离合移动。

```bash
./tools/remote_fetch_best_ground_model.sh
./tools/play_best_ground_model.sh --viewer native --wrist-mode ground
```

使用训练混合分布：`./tools/play_best_ground_model.sh --viewer native`。
强制双腕低位连续运动并运输：
`./tools/play_best_ground_model.sh --viewer native --wrist-mode bilateral`。
`balance` 表示无底盘移动指令，`adjust` 表示世界锚定腕部目标下的小范围移动。
混合模式含静态保持；连续段为接近／抬起／放下，末段保持到回合结束，
下个回合重新采样，不是无限循环轨迹。这是可视化，不是独立评测。

`ground` 每回合采样低位任务，随机低手，目标腕高 0.16--0.28 m；
`mixed` 恢复 balanced 训练分布：25%运输、10%调整、65%自主平衡；
50%高度任务，其中25%低位、低位任务中25%双腕低位；非静态任务中80%
启用连续轨迹。允许机器人移动和自主调整支撑。低位范围是初始锚点范围，
动态抬起段可高于该范围。
可视化保留 play 模式的推力/负载，不等同于无扰动诊断。

按照上述步骤完成后，您已经准备好在虚拟环境中运行相关程序。若遇到问题，请参考各组件的官方文档或检查依赖安装是否正确。
