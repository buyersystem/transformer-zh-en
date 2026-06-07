"""
==============================================================
Transformer 推理脚本
==============================================================

1. 理解推理与训练的区别
2. 掌握Greedy Search和Beam Search的原理
3. 了解模型加载和状态恢复

【推理流程】
加载模型 → 编码源语言 → 解码目标语言 → 输出结果
"""

import torch
import os
import argparse
import sentencepiece as spm
from config import Config
from models.transformer import Transformer
from tokenizer import UnifiedBPETokenizer


def load_checkpoint(checkpoint_path, device):
    """
    加载训练好的模型检查点
    
    【检查点包含】
    - model_state_dict: 模型权重
    - tokenizer: 分词器
    - args: 训练配置
    
    【注意】由于 pickle 无法正确序列化 SentencePiece 对象，
    加载后需要重新加载 BPE 模型文件
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    tokenizer = checkpoint['tokenizer']
    args = checkpoint.get('args', None)
    
    # 重新加载 BPE 模型文件（从 checkpoint 目录自动推导）
    bpe_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    bpe_model_path = os.path.join(bpe_dir, "bpe_unified.model")
    if os.path.exists(bpe_model_path):
        tokenizer.sp = spm.SentencePieceProcessor()
        tokenizer.sp.Load(bpe_model_path)
        tokenizer.model_prefix = os.path.join(bpe_dir, "bpe_unified")
        
        # 同步特殊 token ID
        tokenizer.pad_id = tokenizer.sp.pad_id()
        tokenizer.unk_id = tokenizer.sp.unk_id()
        tokenizer.bos_id = tokenizer.sp.bos_id()
        tokenizer.eos_id = tokenizer.sp.eos_id()
        
        print(f"Reloaded BPE model from {bpe_model_path}")
        print(f"  pad_id={tokenizer.pad_id}, unk_id={tokenizer.unk_id}, bos_id={tokenizer.bos_id}, eos_id={tokenizer.eos_id}")
        print(f"  vocab_size={tokenizer.get_vocab_size()}")
    
    # 兼容旧检查点
    if args is None:
        args = Config()
    
    # 构建模型
    model = Transformer(
        src_vocab_size=len(tokenizer),
        tgt_vocab_size=len(tokenizer),
        d_model=args.d_model,
        num_heads=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        d_ffn=args.d_ff,
        dropout=args.dropout,
        max_len=args.max_len,
        pad_idx=0
    ).to(device)
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # 评估模式，禁用dropout
    
    return model, tokenizer, args


def translate(model, tokenizer, text, device, max_len=100, beam_size=1):
    """
    翻译函数
    
    【参数】
    - model: Transformer模型
    - tokenizer: 分词器
    - text: 待翻译文本
    - device: 计算设备
    - max_len: 最大生成长度
    - beam_size: Beam Search宽度，1表示Greedy Search
    
    【翻译流程】
    1. 判断语言（中文/英文）
    2. 编码源语言
    3. 解码目标语言
    """
    # 判断输入语言，目标语言取反
    is_chinese = any('\u4e00' <= c <= '\u9fff' for c in text)
    src_lang = "zh" if is_chinese else "en"
    tgt_lang = "en" if is_chinese else "zh"
    
    # 编码源语言
    src_ids = tokenizer.encode(text, lang=src_lang, add_bos=False, add_eos=True)
    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)
    
    # 编码器前向传播
    encoder_output, src_mask = model.encode(src_tensor)
    
    # 选择解码策略
    if beam_size == 1:
        return greedy_decode(model, tokenizer, encoder_output, src_mask, device, max_len, tgt_lang)
    else:
        return beam_search_decode(model, tokenizer, encoder_output, src_mask, device, max_len, beam_size, tgt_lang)


def greedy_decode(model, tokenizer, encoder_output, src_mask, device, max_len, tgt_lang):
    """
    贪婪解码 (Greedy Search)
    
    【原理】
    每一步选择概率最高的词
    
    【优点】
    - 简单快速
    
    【缺点】
    - 可能漏掉最优序列
    - 无法回溯
    """
    tgt_ids = [tokenizer.bos_id]  # 从开始符开始
    
    for _ in range(max_len):
        # 解码一步
        tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long).to(device)
        decoder_output = model.decode(tgt_tensor, encoder_output, src_mask)
        output = model.linear(decoder_output)
        
        # 取概率最高的词
        next_token = output[0, -1].argmax().item()
        
        # 遇到结束符停止
        if next_token == tokenizer.eos_id:
            break
        
        tgt_ids.append(next_token)
    
    # 解码为文本
    result_text = tokenizer.decode(tgt_ids, lang=tgt_lang)
    return result_text


def beam_search_decode(model, tokenizer, encoder_output, src_mask, device, max_len, beam_size, tgt_lang):
    """
    Beam Search解码
    
    【原理】
    保留beam_size个最有可能的候选，选择概率乘积最高的
    
    【示例】beam_size=2
    Step 1: ["hello", "hi"] (top 2)
    Step 2: ["hello world", "hello there"] (top 2 of all)
    ...
    
    【优点】
    - 比Greedy更可能找到最优解
    - 考虑多个可能的翻译
    
    【缺点】
    - 速度较慢
    - 内存占用大
    """
    # encoder_output / src_mask 来自单样本 [1, src_len, d_model]，所有 beam 共享
    batch_size = encoder_output.size(0)
    
    # 初始化：每个样本一个候选
    tgt_ids = [[tokenizer.bos_id] for _ in range(batch_size)]
    scores = [0.0] * batch_size
    
    # 存储已完成的候选
    completed = []
    
    for step in range(max_len):
        all_candidates = []
        
        for i in range(len(tgt_ids)):
            # 已完成的跳过
            if tgt_ids[i][-1] == tokenizer.eos_id:
                completed.append((tgt_ids[i], scores[i]))
                continue
            
            # 解码一步（所有 beam 共享同一份 encoder_output）
            tgt_tensor = torch.tensor([tgt_ids[i]], dtype=torch.long).to(device)
            decoder_output = model.decode(tgt_tensor, encoder_output, src_mask)
            output = model.linear(decoder_output)
            
            # 取top-k概率
            probs = torch.softmax(output[0, -1], dim=-1)
            topk_probs, topk_indices = probs.topk(beam_size)
            
            # 扩展候选
            for j in range(beam_size):
                new_seq = tgt_ids[i] + [topk_indices[j].item()]
                new_score = scores[i] + topk_probs[j].item()
                all_candidates.append((new_seq, new_score))
        
        if not all_candidates:
            break
        
        # 按分数排序，保留top-k
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        
        tgt_ids = [c[0] for c in all_candidates[:beam_size]]
        scores = [c[1] for c in all_candidates[:beam_size]]
    
    # 处理未完成的候选
    if not completed:
        completed = [(tgt_ids[0], scores[0])]
    
    # 返回分数最高的
    completed.sort(key=lambda x: x[1], reverse=True)
    result_text = tokenizer.decode(completed[0][0], lang=tgt_lang)
    
    return result_text


def main():
    """
    主函数
    
    【使用方式】
    # 命令行翻译
    python infer.py --input "你好世界"
    
    # 交互式翻译
    python infer.py
    
    # Beam Search
    python infer.py --beam_size 5
    """
    parser = argparse.ArgumentParser(description='Transformer机器翻译推理')
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pt")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--beam_size", type=int, default=1)
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 检查模型文件
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        print("Please run train_llm.py first to train the model.")
        return
    
    # 加载模型
    print(f"Loading checkpoint from {args.checkpoint}...")
    model, tokenizer, config = load_checkpoint(args.checkpoint, device)
    print("Model loaded successfully!")
    
    # 翻译模式
    if args.input:
        # 命令行翻译
        result = translate(model, tokenizer, args.input, device, config.max_len, args.beam_size)
        print(f"Source: {args.input}")
        print(f"Translation: {result}")
    else:
        # 交互式翻译
        print("\n=== Transformer Translation ===")
        print("Enter text to translate (type 'quit' to exit)")
        
        while True:
            text = input("\nSource: ").strip()
            
            if text.lower() == 'quit':
                break
            
            if not text:
                continue
            
            result = translate(model, tokenizer, text, device, config.max_len, args.beam_size)
            print(f"Translation: {result}")


if __name__ == "__main__":
    main()
