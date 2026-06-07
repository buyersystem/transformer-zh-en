# Transformer 中英翻译 — 推理包

纯手写 Transformer 机器翻译模型，即下即用。

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- 显存 4GB+（FP16）/ 无需 GPU（INT8）

## 安装

```bash
pip install torch sentencepiece
```

## 使用方法

```bash
# 默认 FP16 模式（推荐，GPU 推理）
python infer_quantized.py

# INT8 量化模式（CPU 推理）
python infer_quantized.py --model int8

# 单句翻译
python infer_quantized.py --input "你好世界"
python infer_quantized.py --input "Hello world"
```

## 交互式翻译

```
> 你好世界
  hello world
> 今天天气很好
  it 's fine today
> quit
```

输入中文自动译英文，输入英文自动译中文。输入 `quit` 退出。

## 模型信息

| 指标 | 值 |
|------|-----|
| 模型结构 | 4 层 Transformer（手写实现） |
| 参数量 | 25M |
| d_model | 384 |
| 训练数据 | WMT 中英 100 万句对 |
| BLEU (zh→en) | 36.21 |

## 文件说明

```
translation_infer/
├── infer_quantized.py   ← 推理入口
├── models/              ← 模型结构定义
├── checkpoints/         ← 模型权重和分词器
│   ├── model_fp16.pt    ← FP16 权重（102 MB）
│   ├── bpe_unified.*    ← BPE 分词模型
│   └── tokenizer.py     ← 分词器代码
└── README_INFER.md      ← 本文件
```
