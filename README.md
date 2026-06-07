# Transformer 中英机器翻译

纯手写实现 Transformer 论文《Attention Is All You Need》，用于中英机器翻译。

## 项目特性

- **论文原版实现** — 纯手写多头注意力，不使用 `nn.Transformer`
- **统一 BPE 分词** — 中英文共享词表，粒度一致
- **AMP 混合精度** — 自动混合精度训练，节省显存
- **梯度裁剪** — 防止梯度爆炸
- **DDP 多卡支持** — 多卡并行训练
- **TensorBoard 支持** — 实时可视化学习曲线

## 环境要求

| 依赖 | 版本 | 说明 |
|------|------|------|
| Python | 3.8+ | |
| PyTorch | 2.0+ | AMP 混合精度训练 |
| CUDA | 11.0+ | GPU 训练 |
| 显存 | 12GB+ | 4070Ti 单卡 |

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 数据准备

数据未包含在仓库中（体积过大），请从魔搭下载：

👉 [WMT-Chinese-to-English-Machine-Translation-Training-Corpus](https://www.modelscope.cn/datasets/iic/WMT-Chinese-to-English-Machine-Translation-Training-Corpus/files/)

下载后将 CSV 文件放入 `./data/WMT-CN-to-EN/`，然后一键预处理：

```bash
python preprocess_pipeline.py
```

输出：`data/wmt_processed/train.en`、`data/wmt_processed/train.zh`（~100 万句对）

### 2. 训练

```bash
# 推荐配置：4 层轻量模型，~4 小时获得可用翻译
python train_llm.py \
  --data_dir data/wmt_processed \
  --epochs 30 \
  --batch_size 64 \
  --lr_multiplier 0.5 \
  --checkpoint_dir checkpoints \
  --num_encoder_layers 4 \
  --num_decoder_layers 4 \
  --d_model 384 \
  --d_ff 1536
```

多卡训练：
```bash
torchrun --nproc_per_node=3 train_llm.py
```

### 3. 评估

```bash
# 全量评估
python evaluate_bleu.py --checkpoint ./checkpoints/best_model.pt

# 少量样本快速评估
python evaluate_bleu.py --checkpoint ./checkpoints/best_model.pt --max_samples 100
```

| BLEU 分数 | 质量说明 |
|-----------|----------|
| 0-10 | 很差，模型未学习 |
| 10-20 | 一般，基础翻译 |
| 20-30 | 可用，实用级 |
| 30-40 | 较好，商用级 |
| 40+ | 优秀，高质量 |

### 4. 推理

```bash
# 命令行翻译
python infer.py --input "你好世界"

# 交互式翻译
python infer.py

# Beam Search（提高翻译质量）
python infer.py --beam_size 5
```

### 观察训练曲线

```bash
tensorboard --logdir checkpoints/runs
# 浏览器打开 http://localhost:6006
```

## 当前最佳结果

| 指标 | 值 | 说明 |
|------|-----|------|
| val_loss | 2.71 | epoch 11（早停，train-val gap 归零收敛） |
| zh→en BLEU | 36.21 | 1000 样本，greedy decode |
| 总训练时间 | ~4 小时 | 11 epoch，4070Ti 单卡 |
| en→zh | 训练中 | 双向模型，英文→中文仍需更多训练 |

> **早停策略**：train-val gap 在 epoch 11 反转（train < val），此时停止训练可避免过拟合。

### 训练曲线（逐轮记录）

| Epoch | Train Loss | Val Loss | Gap | 备注 |
|-------|-----------|----------|-----|------|
| 0 | 3.42 | 4.14 | -0.72 | 初始快速下降 |
| 1 | 3.19 | 3.26 | -0.07 | |
| 3 | 3.02 | 2.94 | +0.08 | |
| 5 | 2.91 | 2.83 | +0.08 | BLEU=33.55 |
| 8 | 2.78 | 2.76 | +0.02 | 收敛减速 |
| 9 | 2.75 | 2.74 | +0.01 | |
| 10 | 2.72 | 2.72 | **0.00** | 最佳收敛点 |
| 11 | 2.69 | 2.71 | **-0.02** | 过拟合前兆 |

> gap 从正值→零→负值的过程完美展示了过拟合的发生机制。

## 超参数

### 模型结构

| 参数 | 值 | 说明 |
|------|-----|------|
| d_model | 384 | 隐藏层维度（论文原版 512，消费级 GPU 优化） |
| nhead | 8 | 多头注意力头数 |
| num_encoder_layers | 4 | 编码器层数（论文原版 6，小数据防过拟合） |
| num_decoder_layers | 4 | 解码器层数 |
| d_ff | 1536 | 前馈网络维度（4 × d_model） |
| dropout | 0.1 | Dropout 比率 |
| 参数量 | ~25M | 论文原版 65M，缩减至 1/3 |

### 训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| batch_size | 64 | 单卡批次大小 |
| accumulate_grad | 1 | 梯度累积步数 |
| epochs | 30 | 目标训练轮数 |
| warmup_steps | 4000 | 学习率预热步数 |
| lr_multiplier | 0.5 | 学习率乘数（余弦退火） |
| label_smoothing | 0.1 | 标签平滑 |
| clip_grad | 1.0 | 梯度裁剪阈值 |

### 数据配置

| 参数 | 值 | 说明 |
|------|-----|------|
| vocab_size | 32,000 | 中英文统一 BPE 词表 |
| max_len | 128 | 最大序列长度 |
| 训练数据量 | ~100 万句对 | WMT 中英平行语料 |
| 训练耗时 | ~4 小时 | 11 epoch（早停），4070Ti 单卡 |

## 项目结构

```
Transformer_zh_en2026/
├── LICENSE                   # MIT 许可证
├── preprocess_pipeline.py    # 数据预处理流水线
├── train_llm.py              # 训练脚本（AMP + CosineLR + AdamW）【推荐使用】
├── train_2017.py             # 论文原版训练脚本（保留参考）
├── evaluate_bleu.py          # BLEU 评估脚本
├── infer.py                  # 推理脚本（Greedy / Beam Search）
├── quantize.py               # FP16 模型导出脚本
├── infer_quantized.py        # FP16 模型推理脚本
├── config.py                 # 配置参数（含显存适配指南）
├── tokenizer.py              # 统一 BPE 分词器
├── dataset.py                # 数据集加载（TranslationDataset + collate_fn）
├── requirements.txt          # Python 依赖
│
├── models/
│   ├── __init__.py
│   └── transformer.py        # 纯手写 Transformer 实现
│
├── tools/
│   ├── train_tokenizer.py    # 训练 SentencePiece BPE 分词器
│   ├── process_wmt.py        # WMT 原始 CSV → 清洗文本
│   ├── process_subset.py     # 子集数据预处理（去中文空格等）
│   ├── tokenize_text.py      # 分词演示与交互工具
│   ├── test_translate.py     # 翻译功能测试
│   └── check_bpe.py          # BPE 模型信息查看
│
├── scripts/
│   ├── generate_samples.py   # 贪心解码生成
│   ├── generate_beam.py      # Beam Search 生成
│   ├── generate_sampling.py  # 采样生成（temperature / top-k）
│   ├── diagnose_dynamics.py  # 训练动态诊断（lr、loss、grad）
│   ├── analyze_predictions.py# 预测结果分析
│   ├── train_tokenizer_run.py# 分词器训练入口
│   └── ...
│
├── checkpoints/
│   ├── best_model.pt         # 最佳模型检查点（FP32，训练后生成）
│   ├── model_fp16.pt         # FP16 半精度模型（训练后生成）
│   ├── bpe_unified.model     # BPE 分词器模型
│   └── bpe_unified.vocab     # BPE 词表
│
├── translation_infer/        # 独立推理包（可直接发给他人使用）
│   ├── README_INFER.md
│   ├── infer_quantized.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── transformer.py
│   └── checkpoints/
│       ├── bpe_unified.model
│       ├── bpe_unified.vocab
│       └── tokenizer.py
│
├── data/
│   ├── wmt_processed/        # 处理后的训练数据（需下载，不入仓库）
│   └── debug_small/          # 小规模调试数据（200 条训练 + 50 条验证）
│
├── README.md
```

## 显存适配

| GPU | 显存 | batch_size | max_len |
|-----|------|------------|---------|
| RTX 4070Ti | 12GB | 64 | 128 |
| RTX 4090 | 24GB | 64 | 200 |
| 3× RTX 3090 | 24GB×3 | 64 | 200 |

## 核心实现

### 位置编码（论文原版）

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### 多头注意力

- 8 个头，每个头 64 维
- Q、K、V 线性变换
- Scaled dot-product attention

### 学习率调度

**train_llm.py（推荐）**：使用 CosineAnnealingLR + LinearWarmup，训练更稳定。

**train_2017.py（参考）**：论文原版公式
```
lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
```

## 参考

- Attention Is All You Need (Vaswani et al., 2017) — https://arxiv.org/abs/1706.03762

---

## 许可证

本项目采用 MIT 许可证（详见项目根目录 LICENSE 文件）

---

## 使用说明

本项目为书籍《深度学习应用实践————大模型时代的人工智能》配套教学代码，以 MIT 协议开源。

- 可自由使用、修改、分发，包括商用
- 如用于商业培训课程，请注明出处
- 欢迎 Star / Fork / PR
