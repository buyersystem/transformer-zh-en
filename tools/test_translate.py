import sys
import torch
import os
import sentencepiece as spm
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.transformer import Transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load checkpoint
checkpoint = torch.load("./checkpoints/best_model.pt", map_location=device)

# Load tokenizer
from tokenizer import UnifiedBPETokenizer
tokenizer = UnifiedBPETokenizer("./checkpoints/bpe_unified")

# Reload BPE model
tokenizer.sp = spm.SentencePieceProcessor()
tokenizer.sp.Load("./checkpoints/bpe_unified.model")
tokenizer.pad_id = tokenizer.sp.pad_id()
tokenizer.unk_id = tokenizer.sp.unk_id()
tokenizer.bos_id = tokenizer.sp.bos_id()
tokenizer.eos_id = tokenizer.sp.eos_id()

# Load model
args = checkpoint.get('args', None)
vocab_size = len(tokenizer)
model = Transformer(
    src_vocab_size=vocab_size,
    tgt_vocab_size=vocab_size,
    d_model=args.d_model if args else 512,
    num_heads=args.nhead if args else 8,
    num_encoder_layers=args.num_encoder_layers if args else 6,
    num_decoder_layers=args.num_decoder_layers if args else 6,
    d_ffn=args.d_ff if args else 2048,
    dropout=args.dropout if args else 0.2,
    max_len=args.max_len if args else 128,
    pad_idx=0
).to(device)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Test translation
test_cases = ["你好", "我的世界", "hello", "machine translation"]

for text in test_cases:
    is_chinese = any('\u4e00' <= c <= '\u9fff' for c in text)
    src_lang = "zh" if is_chinese else "en"
    tgt_lang = "en" if is_chinese else "zh"
    
    # Encode
    src_ids = tokenizer.encode(text, lang=src_lang, add_bos=False, add_eos=True)
    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)
    
    # Encode
    encoder_output, src_mask = model.encode(src_tensor)
    
    # Greedy decode
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
    print(f"{text} -> {result}")
    print(f"  tgt_ids: {tgt_ids}")