# SSF-GNN：基于光谱—语义图推理的多灾害遥感影像分类

本仓库是论文 **“Towards Efficient Multi-hazard Detection from Single Satellite Imagery via Spectral-Semantic Graph Reasoning”** 的实现。SSF-GNN（Spectral-Semantic Graph Neural Network）以 Sentinel-2 多光谱影像为输入，将光谱指数、深层语义特征和空间邻接关系融合到双分支图推理网络中，用于单幅影像的多灾害分类。

项目同时提供 S2MHD（Sen2MHD）数据组织方式、训练/测试脚本、预训练权重和实验结果。完整的训练过程、混淆矩阵及外部场景推理记录见 [实验记录.md](./实验记录.md)。

## 核心能力

| 项目 | 说明 |
| --- | --- |
| 任务 | 单幅 Sentinel-2 影像的多灾害/非灾害分类 |
| 数据集 | S2MHD（覆盖全球 169 个灾害事件） |
| 输入 | 12 通道 Sentinel-2 GeoTIFF + 20 个光谱指数，共 32 通道 |
| 输出 | 7 类 softmax 分类结果 |
| 模型 | PSPNet-ViG 特征提取 + 语义图/空间图双分支 GATv2 + 三层 MLP |
| 图节点 | 256×256 影像下采样为 8×8，共 64 个节点/样本 |
| 训练 | 单 GPU 或 PyTorch DDP 多 GPU，默认 100 个 epoch |
| 评估 | OA、mIoU、Precision、Recall、F1、F2 和混淆矩阵 |

当前代码中的类别顺序固定为：

`Normal`、`Wildfire`、`Flood`、`Oilspill`、`Redtide`、`Volcaniceruption`、`Algalbloom`。

## 数据集

S2MHD 是基于 Sentinel-2 MSI 影像构建的多灾害数据集，原始样本为包含 12 个波段的 256×256 GeoTIFF。数据集下载地址：[Sen2MHD（百度网盘）](https://pan.baidu.com/s/1Rvr5jYEcW491OiZVjqe5qg?pwd=ppit)。

![S2MHD 数据集示例](Sen2MHD.jpg)

![数据集类别示例](dataset.jpg)

| 属性 | 规格 |
| --- | --- |
| 卫星 | Sentinel-2 MSI |
| 波段 | 12 个蓝/绿/红、红边、NIR、窄 NIR 和 SWIR 波段 |
| 空间分辨率 | 10 m / 20 m（数据集已完成统一重采样） |
| 图像尺寸 | 256 × 256 像素 |
| 文件格式 | GeoTIFF（`.tif`，float32） |
| 类别数 | 7 |
| 数据规模（当前实验） | 41,692 张：训练 20,870，测试 20,822 |

### 目录结构

`dataset.py` 按目录名读取类别，因此目录名和大小写必须与下例一致：

```text
SSF-GNN/
├── README.md
├── requirements.txt
├── run.sh                         # 项目 Python 与 GDAL 启动器
├── train.py                       # 训练（支持单 GPU/DDP）
├── test.py                        # 测试集推理与指标计算
├── model.py                       # SSF-GNN 主模型
├── dataset.py                     # GeoTIFF 数据集与预处理
├── Compute_indices.py             # 20 个光谱指数
├── PSPNet.py / ViG.py             # PSPNet-ViG 主干网络
├── checkpoints_new/               # best.pt、last.pt（本地权重）
├── data/
│   ├── train/
│   │   ├── Normal/
│   │   ├── Wildfire/
│   │   ├── Flood/
│   │   ├── Oilspill/
│   │   ├── Redtide/
│   │   ├── Volcaniceruption/
│   │   └── Algalbloom/
│   └── test/                      # 与 train 相同的类别目录
└── 实验记录.md
```

每个类别目录直接存放 `.tif` 文件，例如 `data/train/Flood/0001.tif`。训练脚本默认使用 `data/train`，验证和测试脚本默认使用 `data/test`；如更换数据位置，请同步修改 `train.py` 或 `test.py` 中的路径。

## 环境配置

已在 Ubuntu、Python 3.12、NVIDIA RTX 5090（驱动 570、CUDA 12.8）上验证。依赖版本见 [requirements.txt](./requirements.txt)：

| 依赖 | 版本/要求 |
| --- | --- |
| PyTorch | 2.9.0 + CUDA 12.8（cu128） |
| PyTorch Geometric | 2.8.0.post1 |
| GDAL | 3.10，提供 `osgeo.gdal` |
| NumPy | 2.5.2 |
| OpenCV | 5.0.0.93 |
| 其他 | matplotlib、tqdm |

### 安装步骤

```bash
cd SSF-GNN

# 1. 创建项目虚拟环境并安装 Python 依赖
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/pip install -r requirements.txt

# 2. 创建提供 GDAL/osgeo 的 Conda 环境（Linux）
conda create -n gdal-env python=3.12 gdal=3.10 \
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/

# 3. 将 GDAL 的 site-packages 写入项目 venv（路径以实际环境为准）
GDAL_SITE=$(conda run -n gdal-env python -c \
  'import site; print(site.getsitepackages()[0])')
echo "$GDAL_SITE" > .venv/lib/python3.12/site-packages/gdal_osgeo.pth
```

`run.sh` 会设置 GDAL 动态库路径并调用 `.venv/bin/python`。脚本中的 Conda 路径默认为 `/home/ubuntu/miniconda3/envs/gdal-env/lib`，在其他机器上请修改该路径，或直接激活对应环境后运行 Python。

安装完成后可执行快速检查：

```bash
./run.sh -c "import torch; from osgeo import gdal; print(torch.__version__); print(torch.cuda.is_available()); print(gdal.VersionInfo())"
```

若使用 RTX 5090 等 Blackwell GPU，请保留 cu128 版本；CUDA 13 构建需要更高版本的 NVIDIA 驱动。

## 数据预处理

每次读取 GeoTIFF 时，`dataset.py` 执行以下处理：

1. 使用 GDAL 读取 12 个波段，将 DN 转换为 `(DN - 1000) / 10000`，并裁剪到 `[-0.1, 1.0]`。
2. 训练阶段随机水平/垂直翻转和 90° 旋转；验证/测试阶段不做随机增强。
3. 使用代码内置的 12 通道均值和标准差标准化原始波段。
4. 在裁剪到 `[0, 1]` 的数据上计算 20 个指数，并与标准化原始波段拼接为 `[B, 32, H, W]`。

20 个指数包括 NDVI、FAI、NDWI、MNDWI（SWIR1/SWIR2）、MGTI、NDOI（SWIR1/SWIR2）、NDBI（SWIR1/SWIR2）、NBR、SAVI、TWI（SWIR1/SWIR2）、CMI、ARVI、NDRE（3 个红边变体）和 RWI。指数计算实现见 [Compute_indices.py](./Compute_indices.py)。

## 模型结构

```text
12 通道影像 ─┐
             ├─ PSPNet-ViG ── 深层语义特征 ─┐
20 个光谱指数 ┘                           │
                                          ├─ 语义图分支 ─ GATv2 ─┐
8×8 节点/边特征 ──────────────────────────┤                      ├─ 拼接 ─ 池化 ─ MLP ─ 7 类
                                          └─ 空间图分支 ─ GATv2 ─┘
```

| 组件 | 配置/作用 |
| --- | --- |
| 主干 | `PSPNet_vig`，ViG tiny 主干提取语义特征 |
| 节点特征 | 主干 384 维特征与指数特征融合，卷积映射到 128 维 |
| 语义图 | 根据 PSPNet 分割预测中同类像素建立连接 |
| 空间图 | 基于 8 邻域边特征建立带边权连接 |
| 图网络 | 两个 GATv2 分支，默认 `GNN_depth=5`、4 头注意力 |
| 分类头 | 三层 MLP（隐藏层 256），输出 7 类 logits |
| 损失 | `CrossEntropyLoss` |
| 参数量 | 约 16.1M（以当前实现统计） |

## 快速开始

### 使用已有权重测试

仓库当前工作区包含 `checkpoints_new/best.pt` 和 `checkpoints_new/last.pt`。若未提供权重，请将训练得到的 `best.pt` 放入该目录；`test.py` 默认加载 `checkpoints_new/best.pt`。

```bash
./run.sh test.py
```

脚本会遍历 `data/test`，输出混淆矩阵、各类别 Precision/Recall/F1/F2 以及总体 OA。推理使用 batch size 64，自动选择 CUDA（可用时）或 CPU。

### 训练

单 GPU：

```bash
./run.sh train.py
```

多 GPU DDP（示例为 2 张 GPU）：

```bash
./run.sh -m torch.distributed.run --standalone \
  --nproc_per_node=2 train.py
```

训练默认配置如下，均在 `train.py` 中定义：

| 配置项 | 默认值 |
| --- | ---: |
| epochs | 100 |
| batch size | 16（每个进程） |
| learning rate | 1e-4 |
| optimizer | Adam |
| 类别数 | 7 |
| GNN depth | 5 |
| 数据增强 | 随机翻转、随机 90° 旋转 |
| DDP 后端 | NCCL |

每轮都会在 `data/test` 上计算验证指标：验证 OA 最优时保存 `checkpoints_new/best.pt`，每轮结束保存 `checkpoints_new/last.pt`。训练日志可重定向保存，项目中的历史日志位于 `ddp_logs/`，曲线数据和图像分别为 `training_curves.csv`、`training_curves.png`。

## 当前实验结果

以下结果来自实验记录中的一组完整训练快照（100 epochs，2×RTX 5090）。最佳权重按验证 OA 选择，随后使用 `checkpoints_new/best.pt` 在当前 `data/test` 划分上评估：

| 指标 | 结果 |
| --- | ---: |
| 训练集/测试集样本数 | 20,870 / 20,822 |
| 最佳验证 OA | 90.207%（第 94 轮） |
| 测试集 OA | **89.953%** |
| 测试集 mIoU | 72.13% |
| Macro-Precision | 83.63% |
| Macro-Recall | 82.81% |
| Macro-F1 | 82.49% |
| Macro-F2 | 82.54% |
| Weighted-F1 | 89.62% |

各类别 F1：Normal 92.83%、Wildfire 91.73%、Flood 89.85%、Oilspill 62.31%、Redtide 92.87%、Volcaniceruption 62.26%、Algalbloom 85.62%。完整混淆矩阵与可视化见 [实验记录.md](./实验记录.md)。

> 注意：当前 `train.py` 使用 `data/test` 做每轮验证，`test.py` 也使用同一目录做最终评估，因此这组结果不等同于严格独立测试，存在一定乐观偏差风险。正式实验建议另行划分验证集和测试集。

## 可选：外部场景推理

仓库还提供针对 Sentinel-2 L2A 外部场景的下载、预处理和切片推理脚本。它们用于生成空间预测可视化，不应替代带标注测试集的定量评估：

```bash
./run.sh prepare_maui_multispectral.py
./run.sh infer_maui_tiles.py
```

相关输入输出路径和实验说明见 [实验记录.md](./实验记录.md)；外部场景不属于 S2MHD 独立标注测试样本。

## 常见问题

- **`from osgeo import gdal` 导入失败**：确认 `gdal-env` 已安装、`.pth` 指向该环境的 `site-packages`，并检查 `run.sh` 中的 `LD_LIBRARY_PATH`。
- **显存不足**：模型包含 ViG 主干和双分支图网络，训练显存开销较大；可在 `train.py` 中减小 batch size，或降低 DDP 的每卡 batch size。
- **类别目录报错**：目录名必须完全匹配七个固定类别，且目录中直接放置 `.tif` 文件。
- **指标出现 NaN**：当某类别在评估批次中没有真实样本或预测样本时，Precision/Recall 可能为 NaN，应结合完整混淆矩阵解读。

## 引用

如果本项目对你的研究有帮助，请引用论文：

```text
Towards Efficient Multi-hazard Detection from Single Satellite Imagery via
Spectral-Semantic Graph Reasoning
```

数据集请引用 S2MHD / Sen2MHD 的原始发布论文，并遵守数据集页面中的使用条款。

## 许可

当前仓库未附带独立 `LICENSE` 文件。代码、模型权重和 S2MHD 数据的再发布与使用请以原论文、数据集发布方及相关依赖的许可证为准。
