"""
Transformer 模型实现（Vaswani et al., 2017）

架构概览：
    src → Embed(+√d) + PE → Encoder(N×) → enc_output ──┐
    tgt → Embed(+√d) + PE → Decoder(N×) → Linear → logits
                                        ↑                |
                                 enc_output ─────────────┘

模型配置（Base版）：
    - d_model: 512
    - num_heads: 8
    - num_encoder/decoder_layers: 6
    - d_ffn: 2048
    - 总参数量: ~93M（含独立 src/tgt embeddings）
"""

import math
import torch
from torch import nn
import torch.nn.functional as F


# ============================================================
# 核心组件
# ============================================================

class PositionalEncoding(nn.Module):
    """
    正弦位置编码（Vaswani et al., Sec 3.5）
    
    为 token embeddings 添加位置信息，弥补纯 attention 架构
    缺乏序列感知能力的不足。使用固定正弦函数：
    
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    正弦函数选择使模型能够通过线性变换学习相对位置关系。
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 预计算位置编码（buffer，不参与梯度更新）
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)   # 偶数维度
        pe[:, 1::2] = torch.cos(position * div_term)   # 奇数维度
        pe = pe.unsqueeze(0)  # [1, max_len, d_model] 便于batch广播
        self.register_buffer('pe', pe)

    def forward(self, x):
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
    缩放点积注意力（Vaswani et al., Eq. 1）
    
        Attention(Q,K,V) = softmax(QK^T / √d_k)V
    
    Args:
        query, key, value: 输入张量 [batch, heads, seq, d_k]
        mask: 可选的布尔掩码，用于padding或因果掩码
    Returns:
        加权值之和及注意力权重
    
    Note: 处理全掩码行产生的NaN，将其替换为零。
    """
    d_k = query.size(-1)
    # 计算注意力分数
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        # 应用掩码：掩码位置填充 -inf
        scores = scores.masked_fill(~mask, float('-inf'))

    # softmax归一化
    attention_weights = F.softmax(scores, dim=-1)

    # 处理全掩码行的NaN
    attention_weights = torch.where(torch.isnan(attention_weights),
                                    torch.zeros_like(attention_weights),
                                    attention_weights)
    # 计算加权和
    return torch.matmul(attention_weights, value), attention_weights


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制（Vaswani et al., Sec 3.2.2）
    
    将输入投影到h个子空间，并行应用注意力，然后拼接并投影回来。
    不同头捕捉关系的不同方面。
    
    计算流程：
        输入 [batch, seq, d_model]
        → 线性投影 (W_q, W_k, W_v, W_o)
        → 拆分为h个头 [batch, h, seq, d_k]
        → 并行缩放点积注意力
        → 拼接 [batch, seq, d_model]
        → 输出投影 W_o
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        # Q, K, V和输出的投影矩阵
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 步骤1: 线性投影并拆分头
        # [batch, seq, d_model] → [batch, h, seq, d_k]
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_v).transpose(1, 2)

        # 步骤2: 每个头并行注意力（掩码自动广播）
        x, attn_weights = scaled_dot_product_attention(Q, K, V, mask)

        # 步骤3: 拼接头并应用输出投影
        # [batch, h, seq, d_k] → [batch, seq, d_model]
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        x = self.dropout(x)
        return self.W_o(x), attn_weights


# ============================================================
# 前馈网络与残差连接
# ============================================================

class PositionWiseFFN(nn.Module):
    """
    逐位置前馈网络（Vaswani et al., Eq. 2）
    
        FFN(x) = ReLU(x·W1 + b1)·W2 + b2
    
    两次线性变换夹ReLU激活：
        d_model(512) → d_ffn(2048) → d_model(512)
    
    在注意力层后独立应用于每个位置。
    """
    def __init__(self, d_model, d_ffn, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ffn),   # 升维
            nn.ReLU(),                     # 非线性激活
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),   # 降维
        )

    def forward(self, X):
        return self.net(X)


class AddNorm(nn.Module):
    """
    残差连接后接层归一化（Vaswani et al., Sec 5.4）
    
        output = LayerNorm(x + Dropout(Sublayer(x)))
    
    使用论文原版的Post-LN架构。
    """
    def __init__(self, normalized_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, residual, sublayer_output):
        return self.ln(residual + self.dropout(sublayer_output))


# ============================================================
# 编码器与解码器层
# ============================================================

class EncoderLayer(nn.Module):
    """
    编码器层，包含自注意力和前馈子层。
    
    架构（自底向上）：
        Input → Self-Attention → Add&Norm → FFN → Add&Norm → Output
    
    每个子层都有残差连接。自注意力使每个token能够关注源序列中的所有其他token。
    """
    def __init__(self, d_model, num_heads, d_ffn, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionWiseFFN(d_model, d_ffn, dropout)
        self.add_norm1 = AddNorm(d_model, dropout)
        self.add_norm2 = AddNorm(d_model, dropout)

    def forward(self, X, mask=None):
        # 子层1: 自注意力（Q=K=V=X，encoder无因果掩码）
        attn_output, _ = self.self_attn(X, X, X, mask)
        X = self.add_norm1(X, attn_output)

        # 子层2: 前馈网络
        ffn_output = self.ffn(X)
        X = self.add_norm2(X, ffn_output)

        return X


class DecoderLayer(nn.Module):
    """
    解码器层，包含掩码自注意力、交叉注意力和FFN。
    
    架构（自底向上）：
        Input → Masked Self-Attn → Add&Norm → Cross-Attn → Add&Norm 
              → FFN → Add&Norm → Output
    
    三个子层：掩码自注意力防止看到未来token，
    交叉注意力关注encoder输出，FFN独立处理。
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
        # 子层1: 掩码自注意力（因果掩码防止前瞻）
        attn_output, _ = self.self_attn(X, X, X, tgt_mask)
        X = self.add_norm1(X, attn_output)

        # 子层2: 交叉注意力（Q=decoder, K=V=encoder输出）
        cross_attn_output, _ = self.cross_attn(X, enc_output, enc_output, src_mask)
        X = self.add_norm2(X, cross_attn_output)

        # 子层3: 前馈网络
        ffn_output = self.ffn(X)
        X = self.add_norm3(X, ffn_output)

        return X


# ============================================================
# 完整Transformer模型
# ============================================================

class Transformer(nn.Module):
    """
    用于序列到序列任务的完整Transformer模型。
    
    数据流：
        src → Embed(×√d) + PE → Encoder(N×) → enc_output ──┐
        tgt → Embed(×√d) + PE → Decoder(N×) → Linear → logits
                                   ↑                  |
                            enc_output (cross-attention)
    
    接口：
        - forward(src, tgt): 训练模式，返回所有logits
        - encode(src): 仅encoder，返回编码输出
        - decode(tgt, ...): 仅decoder，支持逐步生成
    
    Args:
        src_vocab_size, tgt_vocab_size: 词表大小
        d_model: 模型维度（默认512）
        num_heads: 注意力头数（默认8）
        num_encoder_layers: 编码器层数（默认6）
        num_decoder_layers: 解码器层数（默认6）
        d_ffn: FFN隐藏层维度（默认2048）
        dropout: Dropout率（默认0.1）
        max_len: 最大序列长度（默认5000）
        pad_idx: Padding token索引（默认0）
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8,
                 num_encoder_layers=6, num_decoder_layers=6, d_ffn=2048, dropout=0.1,
                 max_len=5000, pad_idx=0):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx

        # Embedding缩放（乘以√d_model以平衡PE）
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)

        # 位置编码（src和tgt共享）
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)

        # 堆叠N层encoder/decoder
        self.encoder = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ffn, dropout)
                          for _ in range(num_encoder_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ffn, dropout)
                          for _ in range(num_decoder_layers)])

        # 输出投影：d_model → vocab_size
        self.linear = nn.Linear(d_model, tgt_vocab_size)

        # Xavier均匀初始化（Vaswani et al., Sec 3.2.2）
        self._init_weights()

    def _init_weights(self):
        """使用Xavier均匀分布初始化权重。"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ============================================================
    # 掩码生成
    # ============================================================

    def generate_square_subsequent_mask(self, sz):
        """
        生成decoder自注意力的因果掩码。
            
        返回下三角矩阵，防止decoder在训练时关注未来token。
            
        示例 (sz=4):
         T T F F  位置0仅见自身
         T T T F  位置1见0,1
         T T T T  位置2见0,1,2
         T T T T  位置3见0,1,2,3
            
        True = 可见，False = 掩码
        """
        mask = torch.tril(torch.ones(sz, sz)).bool()
        return mask

    def make_src_mask(self, src):
        """
        生成源序列padding掩码。
        
        返回 [batch, 1, 1, src_len] 用于注意力中的广播。
        """
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_tgt_mask(self, tgt):
        """
        生成目标序列掩码 = 因果掩码 & padding掩码。
        
        两个条件都为True的位置才可见。
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
        Encoder前向传播。
        
        Args:
            src: 源token IDs [batch, src_len]
        Returns:
            enc_output: 编码表示 [batch, src_len, d_model]
            src_mask: Padding掩码 [batch, 1, 1, src_len]
        """
        src_padding_mask = self.make_src_mask(src)
        src_mask = src_padding_mask & src_padding_mask.transpose(-2, -1)

        # Embedding → 缩放 → 加位置编码
        src_embed = self.src_embed(src) * math.sqrt(self.d_model)
        src_embed = self.pos_encoder(src_embed)

        # 逐层传递通过encoder
        enc_output = src_embed
        for layer in self.encoder:
            enc_output = layer(enc_output, src_mask)
        return enc_output, src_padding_mask

    def decode(self, tgt, encoder_output, src_mask):
        """
        Decoder前向传播，支持逐步生成。
        
        Args:
            tgt: Token IDs [batch, tgt_len]，推理时可追加token
            encoder_output: Encoder输出 [batch, src_len, d_model]
            src_mask: Encoder的padding掩码，用于cross-attention
        
        自动生成因果掩码防止关注未来token。
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
        训练前向传播。
        
        Args:
            src: 源token IDs [batch, src_len]
            tgt: 目标token IDs [batch, tgt_len]
        Returns:
            logits: 输出logits [batch, tgt_len, vocab_size]
        
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
