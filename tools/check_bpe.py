import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.Load('./checkpoints/bpe_unified.model')
print('vocab_size:', sp.get_piece_size())
print('bos_id:', sp.bos_id())
print('eos_id:', sp.eos_id())
print('pad_id:', sp.pad_id())
print('unk_id:', sp.unk_id())
# 测试几个 ID
for i in [0, 1, 2, 3, 100, 1000, 3432, 26211]:
    if i < sp.get_piece_size():
        print(f'{i} -> {sp.id_to_piece(i)}')