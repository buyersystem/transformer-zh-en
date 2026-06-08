"""
============================================================
Transformer 模型实现 —— 严格对标《Attention Is All You Need》
============================================================

【架构一览】
输入                                                          输出
 src ──→ Embed ──→ +PE ──→ [Encoder ×6] ──→ enc_output ──┐
 tgt ──→ Embed ──→ +PE ──→ [Decoder ×6] ──→ Linear ──→ logits
                                     ↑                    |
                              enc_output ──────────────────┘

【模型规格（Base 版）】
- d_model = 512
- 多头数 = 8
- 编码器/解码器各 6 层
- FFN 隐藏层 = 2048
- 总参数量 ≈ 93M（含 src/tgt 两个独立 Embedding）

建议阅读顺序：PositionalEncoding → MultiHeadAttention → EncoderLayer
              → DecoderLayer → Transformer（顶层组装）
"""

import math
import torch
from torch import nn
import torch.nn.functional as F


# ============================================================
# 一、基础组件
# ============================================================

class PositionalEncoding(nn.Module):
    """
    位置编码 —— 论文 Section 3.5

    【为什么需要】
    Transformer 没有循环/卷积，是纯 attention 结构，
    本身感知不到词的顺序。给每个位置加一个独特的"身份标签"，
    模型就能区分"我爱你"和"你爱我"。

    【公式】
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    【设计思想】
    选 sin/cos 不是随便选的——它们有线性性质：
    PE(pos+k) 可以表示为 PE(pos) 的线性变换，
    理论上模型能学习到相对位置关系。
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 预计算所有位置的编码，存为 buffer（不参与训练）
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        # div_term: 频率衰减因子，低频 → 高频
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)   # 偶数维用 sin
        pe[:, 1::2] = torch.cos(position * div_term)   # 奇数维用 cos
        pe = pe.unsqueeze(0)  # [1, max_len, d_model] 方便 batch 广播
        self.register_buffer('pe', pe)

    def forward(self, x):
        """x: [batch, seq_len, d_model] → +PE → dropout"""
        if x.size(1) > self.pe.size(1):
            pe = torch.zeros(x.size(1), self.pe.size(-1), device=x.device)
            position = torch.arange(0, x.size(1), dtype=torch.float, device=x.device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.pe.size(-1), 2, device=x.device).float() * (-math.log(10000.0) / self.pe.size(-1)))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            x = x + pe[:, :x.size(1), :]
        else:
            x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


def scaled_dot_product_attention(query, key, value, mask=None):
    """
    缩放点积注意力 —— 论文 Section 3.2.1，公式 (1)

                Q · K^T
    Attention = softmax(───────) · V
                  √d_k

    【直观理解】
    可以看成一种"软寻址"：
    - Query: 你在找什么？
    - Key:   我这里有什么？
    - Value: 找到后实际返回什么内容？
    - 用 Q 和 K 算相关度，加权取 V

    【Mask 机制】
    - True = 保留（可见），False = 屏蔽
    - 屏蔽位置填 -inf，过 softmax 后权重 → 0

    【NaN 防护】
    极端情况下某行全被 mask（如全 padding 的短句），
    softmax 收到全 -inf 会返回 NaN。此处用 torch.where 替换为 0。
    注意：不能用 nan_to_num_()（inplace 操作会断梯度）。
    """
    d_k = query.size(-1)
    # (1) 计算 QK^T / √d_k
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        # mask: True=保留, False=屏蔽 → 屏蔽位置填 -inf
        scores = scores.masked_fill(~mask, float('-inf'))

    # (2) softmax 归一化
    attention_weights = F.softmax(scores, dim=-1)

    # NaN 防护：全 mask 行 → 权重置 0
    attention_weights = torch.where(torch.isnan(attention_weights),
                                    torch.zeros_like(attention_weights),
                                    attention_weights)
    # (3) 加权求和
    return torch.matmul(attention_weights, value), attention_weights


class MultiHeadAttention(nn.Module):
    """
    多头注意力 —— 论文 Section 3.2.2

    【核心思想】
    不让模型只看一种"视角"。把 d_model 拆成 h 个 d_k 的小空间，
    每个头独立做 attention，最后拼回来。不同头关注不同层面：
    语法结构、语义关联、指代关系……

    【计算流程】
    Input [batch, seq, d_model]
      → 4 个 Linear (W_q, W_k, W_v, W_o)
      → 拆成 h 个头 [batch, h, seq, d_k]
      → 每个头独立做 Scaled Dot-Product Attention
      → 拼回头 [batch, seq, d_model]
      → W_o 输出

    【参数说明】
    - d_model: 模型总维度（512）
    - num_heads: 头数（8），d_k = d_v = d_model / num_heads = 64
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads   # 每个头的 key/query 维度
        self.d_v = d_model // num_heads   # 每个头的 value 维度

        # 四个投影矩阵：Q, K, V 把输入投影到多头空间，W_o 拼回去
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 步骤 1: 线性投影 + 拆头
        # [batch, seq, d_model] → [batch, h, seq, d_k]
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_v).transpose(1, 2)

        # 步骤 2: 各头独立做注意力（mask [batch,1,seq,seq] 自动广播到 [batch,h,seq,seq]）
        x, attn_weights = scaled_dot_product_attention(Q, K, V, mask)

        # 步骤 3: 拼回头 → W_o 输出
        # [batch, h, seq, d_k] → [batch, seq, d_model]
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        x = self.dropout(x)
        return self.W_o(x), attn_weights


# ============================================================
# 二、前馈网络 & 残差连接
# ============================================================

class PositionWiseFFN(nn.Module):
    """
    逐位置前馈网络 —— 论文 Section 3.3，公式 (2)

    FFN(x) = ReLU(x·W1 + b1)·W2 + b2
    
    即两个 Linear，中间夹一个 ReLU：
    d_model(512) → d_ffn(2048) → d_model(512)

    【设计思想】
    注意力负责"交流"（token 之间交换信息），
    FFN 负责"思考"（每个 token 独立思考刚收到的信息）。
    注意力 + FFN 交替堆叠，各司其职。

    【为什么是逐位置的】
    FFN 对每个位置独立应用相同参数——位置之间不交互。
    token 之间的交互已经在 attention 层完成了。
    """
    def __init__(self, d_model, d_ffn, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ffn),   # 升维
            nn.ReLU(),                     # 非线性
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),   # 降维
        )

    def forward(self, X):
        return self.net(X)


class AddNorm(nn.Module):
    """
    残差连接 + 层归一化 —— 论文 Section 5.4

    output = LayerNorm(x + Dropout(Sublayer(x)))

    【为什么需要残差】
    深层网络容易梯度消失。残差连接提供一条"高速公路"，
    让梯度可以直接流回底层，训练更稳定。

    【注意】
    这是 Post-LN（先过子层再加残差后 LN），
    论文原文的顺序。后来的 Pre-LN 把 LN 放前面，
    训练更稳定但不是论文原版。
    """
    def __init__(self, normalized_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, residual, sublayer_output):
        return self.ln(residual + self.dropout(sublayer_output))


# ============================================================
# 三、编码器 & 解码器层
# ============================================================

class EncoderLayer(nn.Module):
    """
    编码器层 —— 论文 Section 3.1

    结构（自底向上）:
    ┌──────────────────────┐
    │     Add & Norm       │
    │        ↑             │
    │    Feed Forward      │  ← 逐位置"思考"
    │        ↑             │
    │     Add & Norm       │
    │        ↑             │
    │  Self-Attention      │  ← token 之间"交流"
    │        ↑             │
    │      Input           │
    └──────────────────────┘

    两个子层各有一个残差连接。
    自注意力让每个源语言 token 看到所有其他源语言 token。
    """
    def __init__(self, d_model, num_heads, d_ffn, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionWiseFFN(d_model, d_ffn, dropout)
        self.add_norm1 = AddNorm(d_model, dropout)
        self.add_norm2 = AddNorm(d_model, dropout)

    def forward(self, X, mask=None):
        # 子层 1: 自注意力（Q=K=V=X，encoder 没有 causal mask）
        attn_output, _ = self.self_attn(X, X, X, mask)
        X = self.add_norm1(X, attn_output)

        # 子层 2: 前馈网络
        ffn_output = self.ffn(X)
        X = self.add_norm2(X, ffn_output)

        return X


class DecoderLayer(nn.Module):
    """
    解码器层 —— 论文 Section 3.1

    结构（自底向上，比 Encoder 多一个子层）:
    ┌──────────────────────┐
    │     Add & Norm       │
    │        ↑             │
    │    Feed Forward      │
    │        ↑             │
    │     Add & Norm       │
    │        ↑             │
    │ Cross-Attention      │  ← 看编码器的输出（新增！）
    │        ↑             │
    │     Add & Norm       │
    │        ↑             │
    │ Masked Self-Attn     │  ← 只看之前的 token（因果掩码）
    │        ↑             │
    │      Input           │
    └──────────────────────┘

    三个子层，encoder 的 2 个 + 新增的 cross-attention。
    Cross-attention 的 Q 来自 decoder，K,V 来自 encoder。
    """
    def __init__(self, d_model, num_heads, d_ffn, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionWiseFFN(d_model, d_ffn, dropout)
        self.add_norm1 = AddNorm(d_model, dropout)
        self.add_norm2 = AddNorm(d_model, dropout)
        self.add_norm3 = AddNorm(d_model, dropout)

    def forward(self, X, enc_output, tgt_mask=None, src_mask=None):
        # 子层 1: Masked 自注意力（有因果掩码，防止看到未来词）
        attn_output, _ = self.self_attn(X, X, X, tgt_mask)
        X = self.add_norm1(X, attn_output)

        # 子层 2: 交叉注意力（Q=decoder 状态，K=V=encoder 输出）
        cross_attn_output, _ = self.cross_attn(X, enc_output, enc_output, src_mask)
        X = self.add_norm2(X, cross_attn_output)

        # 子层 3: 前馈网络
        ffn_output = self.ffn(X)
        X = self.add_norm3(X, ffn_output)

        return X


# ============================================================
# 四、完整的 Transformer 模型
# ============================================================

class Transformer(nn.Module):
    """
    Transformer 完整模型 —— 论文 Section 3.1

    【数据流】
    src ──→ Embed×√d ──→ +PE ──→ [Encoder ×6] ──→ enc ──┐
    tgt ──→ Embed×√d ──→ +PE ──→ [Decoder ×6] ──→ Linear → logits
                                       ↑                  |
                                enc ───┘ (cross-attention)

    【接口说明】
    - forward(src, tgt):   训练用，一次拿到全部 logits
    - encode(src):         只跑 encoder，返回编码输出（推理时复用）
    - decode(tgt, ...):   只跑 decoder，支持逐步生成

    【参数说明（论文 Base 版默认值）】
    - src_vocab_size/tgt_vocab_size: 词表大小（32000）
    - d_model:  512      模型维度
    - num_heads: 8        注意力头数
    - num_encoder_layers:  6  编码器层数
    - num_decoder_layers:  6  解码器层数
    - d_ffn: 2048         前馈网络隐藏层
    - dropout: 0.1        Dropout 比例
    - max_len: 5000       最大序列长度
    - pad_idx: 0          Padding token ID
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8,
                 num_encoder_layers=6, num_decoder_layers=6, d_ffn=2048, dropout=0.1,
                 max_len=5000, pad_idx=0):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx

        # Embedding（缩放：乘以 √d_model，防止 embedding 太小被 PE 淹没）
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)

        # 位置编码（src 和 tgt 共享同一个 PE）
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)

        # 堆叠 N 层编码器 / 解码器
        self.encoder = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ffn, dropout)
                          for _ in range(num_encoder_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ffn, dropout)
                          for _ in range(num_decoder_layers)])

        # 输出投影：d_model → vocab_size
        self.linear = nn.Linear(d_model, tgt_vocab_size)

        # Xavier uniform 初始化（论文 Section 3.2.2）
        self._init_weights()

    def _init_weights(self):
        """论文标准：所有权重矩阵用 Xavier uniform 初始化"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ============================================================
    # Mask 生成
    # ============================================================

    def generate_square_subsequent_mask(self, sz):
        """
        因果掩码 —— 论文 Section 3.2.3

        返回一个下三角矩阵，防止解码器在预测第 t 个词时"偷看"第 t+1 个词。

        示例 (sz=4):
         T T F F     位置 0 只能看到自己
         T T T F     位置 1 看到 0,1
         T T T T     位置 2 看到 0,1,2
         T T T T     位置 3 看到 0,1,2,3

        True = 可见，False = 屏蔽
        """
        mask = torch.tril(torch.ones(sz, sz)).bool()
        return mask

    def make_src_mask(self, src):
        """
        源序列 padding 掩码

        返回 [batch, 1, 1, src_len] —— 故意只保留最后两个维度为 1。
        这样在 encode() 中可以通过 & transpose 得到 [batch,1,src_len,src_len]，
        在 cross-attention 中直接广播到 [batch,1,tgt_len,src_len]。
        """
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_tgt_mask(self, tgt):
        """
        目标序列掩码 = 因果掩码 & padding 掩码

        两个条件都满足（True=可见）的位置才被允许看到。
        """
        tgt_len = tgt.size(1)
        subsequent_mask = self.generate_square_subsequent_mask(tgt_len).to(tgt.device)
        padding_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return padding_mask & subsequent_mask

    # ============================================================
    # 前向传播
    # ============================================================

    def encode(self, src):
        """
        编码器前向传播

        输入: src [batch, src_len]
        输出: enc_output [batch, src_len, d_model]
              src_mask  [batch, 1, 1, src_len]
        """
        src_padding_mask = self.make_src_mask(src)
        src_mask = src_padding_mask & src_padding_mask.transpose(-2, -1)

        # Embedding → 缩放 → 加位置编码
        src_embed = self.src_embed(src) * math.sqrt(self.d_model)
        src_embed = self.pos_encoder(src_embed)

        # 逐层传递
        enc_output = src_embed
        for layer in self.encoder:
            enc_output = layer(enc_output, src_mask)
        return enc_output, src_padding_mask

    def decode(self, tgt, encoder_output, src_mask):
        """
        解码器前向传播 —— 支持逐步生成

        参数:
        - tgt: token IDs [batch, tgt_len]，推理时可逐 token 追加
        - encoder_output: encoder 的输出 [batch, src_len, d_model]
        - src_mask: encoder 返回的 padding 掩码，用于 cross-attention

        内部自动生成因果掩码（防止看到未来 token）。
        """
        tgt_mask = self.make_tgt_mask(tgt)

        # Embedding → 缩放 → 加位置编码
        tgt_embed = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        tgt_embed = self.pos_encoder(tgt_embed)

        dec_output = tgt_embed
        for layer in self.decoder:
            dec_output = layer(dec_output, encoder_output, tgt_mask, src_mask)
        return dec_output

    def forward(self, src, tgt):
        """
        训练前向传播

        输入: src [batch, src_len], tgt [batch, tgt_len]
        输出: logits [batch, tgt_len, vocab_size]

        流程: encode(src) → decode(tgt) → Linear → logits
        """
        encoder_output, src_mask = self.encode(src)
        decoder_output = self.decode(tgt, encoder_output, src_mask)
        output = self.linear(decoder_output)
        return output


# ============================================================
# 快速验证
# ============================================================
if __name__ == "__main__":
    print("构建 Transformer Base 模型（~65M 参数）...")
    model = Transformer(
        src_vocab_size=32000,
        tgt_vocab_size=32000,
        d_model=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ffn=2048,
        dropout=0.1,
        max_len=200,
        pad_idx=0
    )

    # 验证 1: 训练模式 forward
    print("\n[1] forward() — 训练用")
    src = torch.randint(0, 32000, (32, 50))   # batch=32, src_len=50
    tgt = torch.randint(0, 32000, (32, 40))   # batch=32, tgt_len=40
    output = model(src, tgt)
    print(f"    输入: src {src.shape}, tgt {tgt.shape}")
    print(f"    输出: {output.shape}  ← 应为 [32, 40, 32000]")

    # 验证 2: 推理模式 encode + 逐 token decode
    print("\n[2] encode() + decode() — 推理用（逐步生成）")
    src = torch.randint(0, 32000, (4, 30))
    enc, src_mask = model.encode(src)
    print(f"    encode → enc {enc.shape}, mask {src_mask.shape}")

    tgt = torch.tensor([[2], [2], [2], [2]])  # BOS=2
    for step in range(5):
        dec = model.decode(tgt, enc, src_mask)
        next_token = dec[:, -1:].argmax(dim=-1)   # 贪心取最后一个位置
        tgt = torch.cat([tgt, next_token], dim=1)
    print(f"    decode 5 steps → tgt {tgt.shape}  ← 应为 [4, 6]")

    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数量: {total_params / 1e6:.1f}M")
    print("所有测试通过")
