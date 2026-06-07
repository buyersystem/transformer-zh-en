"""Quick inference check on best_model.pt

用法: python scripts/check_inference.py --checkpoint checkpoints/best_model.pt
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import sentencepiece as spm
from models.transformer import Transformer
from tokenizer import UnifiedBPETokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="./checkpoints/best_model.pt", help="模型检查点路径")
cli_args = parser.parse_args()

# 从 checkpoint 路径推导 BPE 目录
bpe_dir = os.path.dirname(os.path.abspath(cli_args.checkpoint))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

checkpoint = torch.load(cli_args.checkpoint, map_location=device)
print(f"Epoch: {checkpoint.get('epoch', '?')}, val_loss: {checkpoint.get('val_loss', 0):.4f}")

tokenizer = UnifiedBPETokenizer(os.path.join(bpe_dir, "bpe_unified"))
if tokenizer.sp is None:
    raise FileNotFoundError(f"BPE model not found in {bpe_dir}")
print(f"Vocab: {tokenizer.get_vocab_size()}, BOS={tokenizer.bos_id}, EOS={tokenizer.eos_id}")

args = checkpoint.get("args", None)
vocab_size = tokenizer.get_vocab_size()
model = Transformer(
    src_vocab_size=vocab_size, tgt_vocab_size=vocab_size,
    d_model=args.d_model if args else 512,
    num_heads=args.nhead if args else 8,
    num_encoder_layers=args.num_encoder_layers if args else 6,
    num_decoder_layers=args.num_decoder_layers if args else 6,
    d_ffn=args.d_ff if args else 2048,
    dropout=args.dropout if args else 0.2,
    max_len=args.max_len if args else 128,
    pad_idx=0
).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Test cases
tests = [
    "你好",
    "我爱编程",
    "今天天气很好",
    "机器学习是人工智能的一个分支",
    "hello",
    "machine learning",
    "how are you",
    "the weather is nice today",
]

for text in tests:
    is_chinese = any("\u4e00" <= c <= "\u9fff" for c in text)
    src_lang = "zh" if is_chinese else "en"
    tgt_lang = "en" if is_chinese else "zh"

    src_ids = tokenizer.encode(text, lang=src_lang, add_bos=False, add_eos=True)
    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)
    encoder_output, src_mask = model.encode(src_tensor)

    tgt_ids = [tokenizer.bos_id]
    for _ in range(100):
        tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long).to(device)
        decoder_output = model.decode(tgt_tensor, encoder_output, src_mask)
        output = model.linear(decoder_output)
        next_token = output[0, -1].argmax().item()
        if next_token == tokenizer.eos_id:
            break
        tgt_ids.append(next_token)

    result = tokenizer.decode(tgt_ids, lang=tgt_lang)
    print(f"[{src_lang}->{tgt_lang}] {text} => {result}")
    print(f"  token_ids: {tgt_ids}")
