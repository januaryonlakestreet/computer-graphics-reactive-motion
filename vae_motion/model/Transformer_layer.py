import os

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
os.sys.path.append(parent_dir)

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Self_Attention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, y, z):
        attn_out, _ = self.attn(x, y, z)
        return self.norm(x + attn_out)

    def create_causal_mask(self,size):
        """
        Creates a causal mask for a transformer.

        Args:
          size: The size of the sequence.

        Returns:
          A tensor representing the causal mask.
        """
        mask = torch.tril(torch.ones(size, size))
        return mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))


class Cross_Attention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, y, z):
        attn_out, _ = self.attn(x, y, z)
        return self.norm(x + attn_out)


    def create_causal_mask(self,size_a,side_b):
        """
        Creates a causal mask for a transformer.

        Args:
          size: The size of the sequence.

        Returns:
          A tensor representing the causal mask.
        """
        mask = torch.tril(torch.ones(size_a, side_b))
        return mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))






    def create_causal_mask(self,size_a,side_b):
        """
        Creates a causal mask for a transformer.

        Args:
          size: The size of the sequence.

        Returns:
          A tensor representing the causal mask.
        """
        mask = torch.tril(torch.ones(size_a, side_b))
        return mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

class FeedForward(nn.Module):
    def __init__(self, dim,dim_mid, dim_out, dropout, activation):
        super().__init__()

        self.layer1 = nn.Linear(dim, dim_mid)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.layer2 = nn.Linear(dim_mid, dim_out)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x







class DecoderTransformerLayer(nn.Module):
    def __init__(self, latent_dim,num_heads=4, dropout=0.1, ff_dim=1024):


        super().__init__()

        self.self_attn = Self_Attention(latent_dim, num_heads, 0.1)
        self.cross_attn = Cross_Attention(latent_dim, num_heads, 0.1)
        self.ffn = FeedForward(latent_dim,ff_dim,latent_dim,dropout,nn.GELU())

        self.final_out = nn.Linear(latent_dim,latent_dim)

        self.layernorm = nn.LayerNorm(1024)





    def forward(self, transformer_in,memory):
        t_in = self.self_attn(transformer_in, transformer_in, transformer_in)
        transformer_in = self.layernorm(t_in + transformer_in)
        transformer_in = self.ffn(transformer_in)

        t_in = self.cross_attn(transformer_in, memory, memory)
        transformer_in = self.layernorm(t_in + transformer_in)
        transformer_in = self.ffn(transformer_in)


        return self.final_out(transformer_in)

