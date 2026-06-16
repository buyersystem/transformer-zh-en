"""
=============================================================
BPE 分词工具脚本
=============================================================

对文本进行 encode/decode。需要先训练 BPE 模型。
用法: python tokenize_text.py
"""

import sentencepiece as spm
import sys


def load_tokenizer(model_path="./checkpoints/bpe_unified.model"):
    """加载 BPE 分词器"""
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)
    print(f"加载分词器: {model_path}")
    print(f"词表大小: {sp.get_piece_size()}")
    return sp


def tokenize_text(sp, text):
    """
    分词函数
    
    参数:
        sp: SentencePiece 处理器
        text: 待分词文本
    
    返回:
        ids: token ID 列表
        pieces: 子词列表
    """
    # 编码
    ids = sp.Encode(text, add_bos=True, add_eos=True)
    pieces = [sp.id_to_piece(i) for i in ids]
    
    return ids, pieces


def decode_text(sp, ids):
    """
    解码函数
    
    参数:
        sp: SentencePiece 处理器
        ids: token ID 列表
    
    返回:
        text: 解码后的文本
    """
    text = sp.Decode(ids)
    return text


def demo_tokenization():
    """演示分词器使用"""
    # 加载分词器
    sp = load_tokenizer()
    
    print("\n" + "=" * 50)
    print("分词演示")
    print("=" * 50)
    
    # 中文示例
    print("\n【中文分词】")
    zh_text = "机器翻译是人工智能的重要应用"
    zh_ids, zh_pieces = tokenize_text(sp, zh_text)
    print(f"原文: {zh_text}")
    print(f"IDs: {zh_ids}")
    print(f"子词: {zh_pieces}")
    
    # 解码
    decoded = decode_text(sp, zh_ids)
    print(f"解码: {decoded}")
    
    # 英文示例
    print("\n【英文分词】")
    en_text = "machine translation is an important application of artificial intelligence"
    en_ids, en_pieces = tokenize_text(sp, en_text)
    print(f"原文: {en_text}")
    print(f"IDs: {en_ids}")
    print(f"子词: {en_pieces}")
    
    # 解码
    decoded = decode_text(sp, en_ids)
    print(f"解码: {decoded}")


def interactive_mode():
    """交互模式"""
    sp = load_tokenizer()
    
    print("\n" + "=" * 50)
    print("交互分词模式")
    print("=" * 50)
    print("输入文本进行分词，输入 q 退出")
    print("-" * 50)
    
    while True:
        text = input("\n请输入文本: ").strip()
        if text.lower() == 'q':
            break
        
        # 判断语言
        is_chinese = any('\u4e00' <= c <= '\u9fff' for c in text)
        lang = "中文" if is_chinese else "英文"
        
        # 分词
        ids, pieces = tokenize_text(sp, text)
        
        print(f"语言: {lang}")
        print(f"Token 数量: {len(ids)}")
        print(f"子词: {pieces}")
        
        # 解码验证
        decoded = decode_text(sp, ids)
        print(f"解码: {decoded}")


def main():
    # 如果有命令行参数，使用参数作为输入文本
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        sp = load_tokenizer()
        ids, pieces = tokenize_command_arguments(sp, text)
        print(f"分词结果: {pieces}")
        print(f"Token IDs: {ids}")
    else:
        # 演示模式
        demo_tokenization()
        
        # 交互模式
        print("\n是否进入交互模式? (y/n): ", end="")
        if input().strip().lower() == 'y':
            interactive_mode()


def tokenize_command_arguments(sp, text):
    """处理命令行参数的分词"""
    is_chinese = any('\u4e00' <= c <= '\u9fff' for c in text)
    ids = sp.Encode(text, add_bos=True, add_eos=True)
    pieces = [sp.id_to_piece(i) for i in ids]
    return ids, pieces


if __name__ == "__main__":
    main()