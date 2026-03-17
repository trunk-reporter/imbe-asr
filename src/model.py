"""Conformer-CTC model for IMBE-to-text speech recognition.

Architecture: Linear input projection -> Conformer encoder -> CTC head.
9.2M parameters at d_model=256, n_layers=6, d_ff=1024.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """Conformer feed-forward module with Swish activation."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.ln(x)
        x = self.linear1(x)
        x = x * torch.sigmoid(x)  # Swish
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class ConvModule(nn.Module):
    """Conformer convolution module."""

    def __init__(self, d_model, kernel_size=31, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.pw1 = nn.Linear(d_model, 2 * d_model)  # pointwise + GLU
        self.dw = nn.Conv1d(
            d_model, d_model, kernel_size,
            padding=kernel_size // 2, groups=d_model,
        )
        self.bn = nn.BatchNorm1d(d_model)
        self.pw2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, D)
        x = self.ln(x)
        x = self.pw1(x)
        x = F.glu(x, dim=-1)               # (B, T, D)
        x = x.transpose(1, 2)              # (B, D, T) for conv
        x = self.dw(x)
        x = self.bn(x)
        x = x.transpose(1, 2)              # (B, T, D)
        x = x * torch.sigmoid(x)           # Swish
        x = self.pw2(x)
        x = self.dropout(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention using F.scaled_dot_product_attention.

    Uses manual Q/K/V projections with a fused linear layer and calls
    F.scaled_dot_product_attention which auto-dispatches to FlashAttention-2
    or memory-efficient attention on PyTorch 2.0+.
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout_p = dropout

        self.ln = nn.LayerNorm(d_model)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        x = self.ln(x)
        B, T, D = x.shape

        # Fused Q/K/V projection and reshape to (B, n_heads, T, head_dim)
        qkv = self.qkv_proj(x)  # (B, T, 3*D)
        qkv = qkv.view(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, n_heads, T, head_dim)
        q, k, v = qkv.unbind(0)  # each (B, n_heads, T, head_dim)

        # Build attention mask from key_padding_mask
        # key_padding_mask: (B, T) bool, True = pad position
        attn_mask = None
        if key_padding_mask is not None:
            # Expand to (B, 1, 1, T) for broadcast over heads and query positions
            # True (pad) -> -inf, False (valid) -> 0.0
            attn_mask = torch.zeros(B, 1, 1, T, dtype=q.dtype, device=q.device)
            attn_mask.masked_fill_(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
        )  # (B, n_heads, T, head_dim)

        # Merge heads and project
        out = out.transpose(1, 2).contiguous().view(B, T, D)  # (B, T, D)
        out = self.out_proj(out)
        return self.dropout(out)


class ConformerBlock(nn.Module):
    """Single Conformer block: FFN(1/2) -> MHSA -> Conv -> FFN(1/2) -> LayerNorm."""

    def __init__(self, d_model, n_heads, d_ff, conv_kernel=31, dropout=0.1):
        super().__init__()
        self.ff1 = FeedForward(d_model, d_ff, dropout)
        self.mhsa = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.conv = ConvModule(d_model, conv_kernel, dropout)
        self.ff2 = FeedForward(d_model, d_ff, dropout)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x, key_padding_mask=None):
        x = x + 0.5 * self.ff1(x)
        x = x + self.mhsa(x, key_padding_mask=key_padding_mask)
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        x = self.ln(x)
        return x


class SinusoidalPE(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, D)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class ConformerCTC(nn.Module):
    """Conformer encoder with CTC head for IMBE-to-text.

    Args:
        input_dim: Input feature dimension (170 for raw IMBE params)
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward inner dimension
        n_layers: Number of Conformer blocks
        conv_kernel: Convolution kernel size
        vocab_size: Output vocabulary size (including blank)
        dropout: Dropout rate
        subsample: If True, apply 2x strided conv to reduce frame rate
    """

    def __init__(self, input_dim=170, d_model=256, n_heads=4, d_ff=1024,
                 n_layers=6, conv_kernel=31, vocab_size=40, dropout=0.1,
                 subsample=False):
        super().__init__()
        self.subsample = subsample

        if subsample:
            # 2x strided convolution: 50fps -> 25fps
            self.subsample_conv = nn.Sequential(
                nn.Conv1d(input_dim, d_model, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.LayerNorm(d_model),
            )
        else:
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, d_model),
                nn.LayerNorm(d_model),
                nn.Dropout(dropout),
            )

        self.pos_enc = SinusoidalPE(d_model)

        self.encoder = nn.ModuleList([
            ConformerBlock(d_model, n_heads, d_ff, conv_kernel, dropout)
            for _ in range(n_layers)
        ])

        self.ctc_head = nn.Linear(d_model, vocab_size)

    def forward(self, x, input_lengths=None):
        """
        Args:
            x: (B, T, input_dim) IMBE features
            input_lengths: (B,) actual frame counts (before padding)

        Returns:
            log_probs: (B, T', vocab_size) log probabilities
            output_lengths: (B,) actual output frame counts
        """
        if self.subsample:
            x = x.transpose(1, 2)           # (B, D_in, T)
            x = self.subsample_conv(x)
            x = x.transpose(1, 2)           # (B, T//2, D)
            if input_lengths is not None:
                output_lengths = (input_lengths + 1) // 2
            else:
                output_lengths = None
        else:
            x = self.input_proj(x)          # (B, T, D)
            output_lengths = input_lengths

        x = self.pos_enc(x)

        # Build padding mask for attention
        key_padding_mask = None
        if output_lengths is not None:
            T = x.size(1)
            key_padding_mask = torch.arange(T, device=x.device).unsqueeze(0) >= \
                output_lengths.unsqueeze(1)  # (B, T) True=pad

        for block in self.encoder:
            x = block(x, key_padding_mask=key_padding_mask)

        logits = self.ctc_head(x)           # (B, T, V)
        log_probs = F.log_softmax(logits, dim=-1)

        return log_probs, output_lengths

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
