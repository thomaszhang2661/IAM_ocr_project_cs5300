# 训练过程记录

CS5300 Final Project — IAM 手写识别 CRNN 训练流水账

---

## 项目目标

用 CRNN 模型做两个对比实验：
1. **Experiment 1**：全量 IAM 数据训练 → 测试
2. **Experiment 2**：剔除 VLM 标注质量有问题的样本后训练 → 测试

数据集：HuggingFace `Teklia/IAM-line`，train=6482 / val=976 / test=2915 行。

---

## 环境准备

- 新建 conda 环境 `ocr_IAM`（原 yolo 环境依赖不兼容）
- 重写 `data/prepare_lmdb.py`：从 HF CSV + 图片格式读取，转存为 LMDB（key-value，fast IO）
- 图片统一 resize 到 H=64（原始 H=32 太小，特征丢失）
- LMDB 目录：`data/lmdb/train/`, `val/`, `test/`，后续还会生成 `train_clean/`

---

## 模型

### v1：VGG + BiLSTM（`htr_model/model.py`）
- 参考原始 CRNN 论文架构
- 最终没有作为主力实验，仅验证可用

### v2：VGG + BiGRU（`htr_model/model_v2.py`）
- 按照参考脚本 `tools/train_crnn.py` + `tools/config.yaml` 重新实现
- VGG backbone：7层 CNN，输出 512 通道
- Encoder：BiGRU，hidden_size=256，2层，双向
- 高度坍缩：`mean(dim=2)`（对 H=64 时 h>1 更鲁棒，比 squeeze 稳）
- 初始参数量：**11,103,568（约 11M）**
- 关键修复：最后一层 conv kernel 从固定 2 改为 `img_h // 16`，支持 H=32/64

---

## 图像增强（`htr_model/augment.py`）

在线增强（每个 epoch 随机生成，不存盘），参考 `tools/image_utils.py`：

| 操作 | 参数 |
|------|------|
| Affine（缩放+剪切+平移） | sx/sy=0.05, shear=0.75, tx/ty=0.01 |
| Gamma 校正 | [0.001, 1.0]（对应 config RANDOM_GAMMA=1） |
| Gaussian blur | sigma [0.3, 1.0] |
| Gaussian noise | std=0.03 |
| Salt & Pepper | amount=0.008 |
| Elastic distortion | alpha=6.0, sigma=3.0 |

**踩坑**：Affine 变换最初直接 warp 导致图像内容被截断。修复方法：前向映射四个角点 → 计算 bounding box → 扩展 canvas → 再 resize 回原始 H。

---

## 超参数对齐（参考 `tools/config.yaml`）

| 参数 | 原始值 | 对齐后 |
|------|--------|--------|
| lr | 1e-3 | **1e-4** |
| warmup_epochs | — | **1** |
| clip_grad | 5.0 | **10.0** |
| gamma range | [0.5, 1.5] | **[0.001, 1.0]** |
| CTC reduction | mean | mean（sum/N 数值不稳定，放弃） |
| plateau patience | 0 | **5 → 10**（见下） |
| plateau factor | 0.8 | 0.8 |
| plateau cooldown | 0 | 2 |
| weight_decay | — | 1e-4 |
| batch_size | — | 64 |

---

## 多卡尝试（失败）

服务器有 8× H100 80GB，尝试 DataParallel 4卡加速。

**问题**：VGG backbone 有 BatchNorm2d。DataParallel 把 batch=64 拆成每卡 16，BatchNorm 统计量在小 batch 下极不稳定，训练无法收敛（epoch 12 后 CER 仍 84%+）。

**结论**：放弃多卡，单 H100 已足够（每 epoch 约 12-14s，100 epoch ≈ 17 分钟）。

---

## 训练记录

### Run 1：`original_v2`（参数未完全对齐，lr=1e-3，patience=0）
- 多次被中断重启，log 混杂
- 最终 100 epoch，best val_CER = **7.80%**

### Run 2：`original_v2`（参数完全对齐，lr=1e-4，patience=5）
- 100 epoch，best val_CER = **8.00%**
- 分析：patience=5 过小，val_CER 在增强下自然抖动 ±1%，scheduler 误判为停滞，LR 从 1e-4 连续降至 2.62e-5，模型过早陷入局部最优
- 最终 lr 只剩 2-3e-5，已无法继续优化

### Run 3：`original_v2_r2`（patience=10，cooldown=2）—— 已中断
- 刚起步就因决定同时修改 hidden_size 而中断

### Run 4（当前）：`original_v2_h128`
- hidden_size: 256 → **128**（参数 11.1M → 9.5M）
- 新增 dropout=0.1（CNN 输出后 + GRU 层间）
- patience=10，cooldown=2，150 epoch
- 进行中……

---

## 关于减小模型的分析

- CNN 骨干（512 通道）占 ~8M 参数，VGG 卷积权重在空间上共享，实际冗余度低，**不建议减少 CNN 输出通道**
- BiGRU hidden_size 256→128 对序列建模影响可接受
- 若需进一步减参，可在 CNN→GRU 之间加 `Linear(512→256)` 投影层，保留 CNN 特征提取能力的同时减小 GRU 规模

---

## 待完成

- [ ] Run 4 结束后记录 val_CER
- [ ] 生成 `train_clean` LMDB（用 VLM 标注结果剔除问题样本）
- [ ] Experiment 2：clean 模式训练
- [ ] `evaluate_iam.py` 分别跑 test set，得到最终 CER/WER 对比

---

## 文件结构

```
data/
  lmdb/train/, val/, test/        — 全量 LMDB
  lmdb/train_clean/               — 待生成（clean 实验用）

htr_model/
  model.py                        — v1: VGG+BiLSTM
  model_v2.py                     — v2: VGG+BiGRU（主力）
  dataset.py                      — LMDBDataset + Converter
  augment.py                      — 在线图像增强

train_iam.py                      — 训练主脚本（支持 --resume）
evaluate_iam.py                   — 测试脚本

checkpoints/
  original_v2/best.pt             — Run 2 最佳模型（val CER 8.00%）
  original_v2_h128/               — Run 4 进行中

logs/
  train_original_v2.log
  train_original_v2_h128.log
```
