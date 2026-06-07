"""数据质量、分词器和数据集检查"""
import os
import sys
import random
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_raw_data():
    print("="*60)
    print("原始数据质量")
    print("="*60)
    
    with open('data/wmt_processed/train.zh', 'r', encoding='utf-8') as f:
        zh_lines = f.readlines()
    with open('data/wmt_processed/train.en', 'r', encoding='utf-8') as f:
        en_lines = f.readlines()
    
    print(f"句对总数: {len(zh_lines)}")
    print(f"中文空行: {sum(1 for l in zh_lines if l.strip()=='')}")
    print(f"英文空行: {sum(1 for l in en_lines if l.strip()=='')}")
    
    zh_lens = [len(l.strip()) for l in zh_lines]
    en_lens = [len(l.strip()) for l in en_lines]
    print(f"\n中文长度 - 最小: {min(zh_lens)}, 最大: {max(zh_lens)}, 平均: {sum(zh_lens)/len(zh_lens):.1f}")
    print(f"英文长度 - 最小: {min(en_lens)}, 最大: {max(en_lens)}, 平均: {sum(en_lens)/len(en_lens):.1f}")
    
    print("\n--- 前3个样本 ---")
    for i in range(3):
        print(f"[{i}] ZH: {zh_lines[i].strip()[:100]}")
        print(f"[{i}] EN: {en_lines[i].strip()[:100]}")
    
    print("\n--- 随机3个样本 ---")
    random.seed(42)
    for i in random.sample(range(len(zh_lines)), 3):
        print(f"[{i}] ZH: {zh_lines[i].strip()[:100]}")
        print(f"[{i}] EN: {en_lines[i].strip()[:100]}")
    
    very_long = sum(1 for z,e in zip(zh_lens, en_lens) if z>500 or e>500)
    very_short = sum(1 for z,e in zip(zh_lens, en_lens) if z<5 or e<5)
    print(f"\n超长句对(>500字符): {very_long}")
    print(f"超短句对(<5字符): {very_short}")
    
    # 检查中英文比例异常
    ratio_issues = sum(1 for z,e in zip(zh_lens, en_lens) if z>0 and e>0 and (z/e > 10 or e/z > 10))
    print(f"长度比例异常(>10倍): {ratio_issues}")

def check_tokenizer():
    print("\n" + "="*60)
    print("分词器检查")
    print("="*60)
    
    from tokenizer import build_tokenizer
    
    tokenizer = build_tokenizer(
        'data/wmt_processed/train.zh',
        'data/wmt_processed/train.en',
        32000,
        'checkpoints/bpe_unified'
    )
    
    print(f"词表大小: {len(tokenizer)}")
    print(f"PAD ID: {tokenizer.pad_id}, EOS ID: {tokenizer.eos_id}, BOS ID: {tokenizer.bos_id}")
    
    # 测试编解码
    test_zh = "你好世界，这是一个测试。"
    test_en = "Hello world, this is a test."
    
    zh_ids = tokenizer.encode(test_zh, lang="zh", add_bos=False, add_eos=True)
    en_ids = tokenizer.encode(test_en, lang="en", add_bos=True, add_eos=True)
    
    print(f"\n中文测试: '{test_zh}'")
    print(f"  编码: {zh_ids}")
    print(f"  解码: '{tokenizer.decode(zh_ids)}'")
    
    print(f"\n英文测试: '{test_en}'")
    print(f"  编码: {en_ids}")
    print(f"  解码: '{tokenizer.decode(en_ids)}'")
    
    # 检查roundtrip
    zh_roundtrip = tokenizer.decode(zh_ids)
    en_roundtrip = tokenizer.decode(en_ids)
    print(f"\nRoundtrip 中文: {'PASS' if zh_roundtrip.strip() == test_zh.strip() else 'FAIL'}")
    print(f"Roundtrip 英文: {'PASS' if en_roundtrip.strip() == test_en.strip() else 'FAIL'}")
    
    # 统计训练数据token长度
    with open('data/wmt_processed/train.zh', 'r', encoding='utf-8') as f:
        zh_lines = f.readlines()
    with open('data/wmt_processed/train.en', 'r', encoding='utf-8') as f:
        en_lines = f.readlines()
    
    zh_token_lens = []
    en_token_lens = []
    for i in range(min(1000, len(zh_lines))):
        z = tokenizer.encode(zh_lines[i].strip(), lang="zh", add_bos=False, add_eos=True)
        e = tokenizer.encode(en_lines[i].strip(), lang="en", add_bos=True, add_eos=True)
        zh_token_lens.append(len(z))
        en_token_lens.append(len(e))
    
    print(f"\nToken长度统计(抽样1000句):")
    print(f"  中文 - 最小: {min(zh_token_lens)}, 最大: {max(zh_token_lens)}, 平均: {sum(zh_token_lens)/len(zh_token_lens):.1f}")
    print(f"  英文 - 最小: {min(en_token_lens)}, 最大: {max(en_token_lens)}, 平均: {sum(en_token_lens)/len(en_token_lens):.1f}")
    print(f"  超过128的: 中文{sum(1 for x in zh_token_lens if x>128)}句, 英文{sum(1 for x in en_token_lens if x>128)}句")

def check_dataset():
    print("\n" + "="*60)
    print("训练数据集检查")
    print("="*60)
    
    from tokenizer import build_tokenizer
    from dataset import TranslationDataset, collate_fn
    from torch.utils.data import DataLoader
    
    tokenizer = build_tokenizer(
        'data/wmt_processed/train.zh',
        'data/wmt_processed/train.en',
        32000,
        'checkpoints/bpe_unified'
    )
    
    dataset = TranslationDataset('data/wmt_processed', tokenizer, max_len=128, split="train")
    
    print(f"数据集大小: {len(dataset)}")
    
    # 检查单个样本
    sample = dataset[0]
    print(f"\n样本0:")
    print(f"  src shape: {sample['src'].shape}, tgt shape: {sample['tgt'].shape}")
    print(f"  src: {sample['src'][:20].tolist()}...")
    print(f"  tgt: {sample['tgt'][:20].tolist()}...")
    
    # 检查DataLoader
    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    batch_src, batch_tgt = next(iter(loader))
    print(f"\nBatch检查 (batch_size=32):")
    print(f"  src batch shape: {batch_src.shape}")
    print(f"  tgt batch shape: {batch_tgt.shape}")
    print(f"  src padding比例: {(batch_src == 0).sum().item() / batch_src.numel() * 100:.1f}%")
    print(f"  tgt padding比例: {(batch_tgt == 0).sum().item() / batch_tgt.numel() * 100:.1f}%")
    
    # 检查EOS位置
    eos_mask = batch_tgt == tokenizer.eos_id
    print(f"  每句有EOS: {eos_mask.any(dim=1).all().item()}")
    
    # 检查BOS位置  
    bos_mask = batch_tgt == tokenizer.bos_id
    print(f"  每句有BOS: {bos_mask.any(dim=1).all().item()}")

if __name__ == "__main__":
    check_raw_data()
    check_tokenizer()
    check_dataset()
    print("\n" + "="*60)
    print("检查完成")
    print("="*60)
