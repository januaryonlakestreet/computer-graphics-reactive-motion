import os

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
os.sys.path.append(parent_dir)

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
import torch.optim as optim



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

class MotionEncoder(nn.Module):
    def __init__(self, motion_input_size, hidden_dim, latent_dim, num_layers=7, num_heads=4, dropout=0.1, ff_dim=1024, sequence_length=32):
        super().__init__()
        self.motion_proj = nn.Sequential(nn.Linear(motion_input_size, hidden_dim),nn.LayerNorm(hidden_dim))
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        encoder_layer = TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout, activation='gelu', batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)


        self.mean_proj = nn.Linear(latent_dim, latent_dim*2)
        self.log_var_proj = nn.Linear(latent_dim, latent_dim*2)
        self.sequence_length = sequence_length

        self.global_motion_token = nn.Parameter(
            torch.randn(2, hidden_dim))
        self.latent_dim = latent_dim
        self.encoded_motion_to_latent = nn.Linear(hidden_dim,latent_dim)
        self.hd = hidden_dim


    def forward(self, motion_primitive):

        bs = motion_primitive.shape[0]
        motion_src = self.motion_proj(motion_primitive)

        dist = torch.tile(self.global_motion_token[:, None, :], (1, bs, 1))
        enc_vector = torch.cat((motion_src,dist.permute(1,0,2)),dim=1)

        motion_encoded = self.transformer_encoder(self.pos_encoder(enc_vector))
        motion_encoded = self.encoded_motion_to_latent(motion_encoded)

        mean = self.mean_proj(motion_encoded[:,1,:])
        log_var = self.log_var_proj(motion_encoded[:,2,:])

        return mean, log_var




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

class MotionDecoder(nn.Module):
    def __init__(self, latent_dim,  hidden_dim, motion_output_size,
                 num_layers=7, num_heads=4, dropout=0.1, ff_dim=1024,
                 history_length=5, future_length=1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)


        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)

        self.new_transformer_decoder = nn.ModuleList([
            DecoderTransformerLayer(hidden_dim, num_heads,dropout,ff_dim) for _ in range(num_layers)
        ])

        self.motion_projection = nn.Linear(int(motion_output_size/2), hidden_dim)
        self.history_length = history_length
        self.future_length = future_length
        self.output_proj = nn.Linear(hidden_dim, int(motion_output_size/2))

        # Make sure embedding dimension matches hidden_dim
        self.future_pos_emb = nn.Embedding(future_length, hidden_dim)
        self.future_start_token = nn.Parameter(torch.randn(1, 1, hidden_dim))  # Learned start token

        self.history_positions_projection = nn.Linear(66,1024)

    def forward(self, z, history_frames):
        self.history_length = history_frames.shape[1]
        # 1. Base latent processing
        memory = self.latent_proj(z).unsqueeze(1)

        # 3. History embedding (specialized)
        history_embed = self.motion_projection(history_frames)

        history_embed = self.pos_encoder(history_embed)

        memory = self.pos_encoder(memory)

        # 6. Transformer processing
        layer_in = memory
        for layer in self.new_transformer_decoder:
            layer_in = layer(layer_in, history_embed)

        # 7. Final projection
        return self.output_proj(layer_in)






class DartVAE(nn.Module):
    def __init__(self, motion_input_size,  hidden_dim=1024, latent_dim=256, motion_output_size=None, sequence_length=32, num_layers=7, num_heads=4, dropout=0.1, ff_dim=2048,normalization=None,history_length=10):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



        self.motion_encoder = MotionEncoder(motion_input_size=motion_input_size,
                                               hidden_dim=hidden_dim,
                                               latent_dim=int(latent_dim/2),
                                               num_layers=num_layers,
                                               num_heads=num_heads,
                                               dropout=dropout,
                                               ff_dim=ff_dim,
                                               sequence_length=sequence_length).to(self.device)



        self.motion_decoder = MotionDecoder(latent_dim=latent_dim,
                                               hidden_dim=hidden_dim,
                                               motion_output_size=motion_output_size,
                                               num_layers=7,
                                               num_heads=4,
                                               dropout=dropout,
                                               ff_dim=ff_dim,
                                               history_length=1,
                                               future_length=1).to(self.device)






        self.latent_dim = latent_dim
        self.optim = optim.Adam(list(self.motion_encoder.parameters())+list(self.motion_decoder.parameters()), lr=0.00001)






    def encode(self, motion_primitive):
        mean, log_var = self.motion_encoder(motion_primitive)
        return {
                "mean":mean,
                "log_var":log_var,
                }


    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std



    def decode(self, z,motion_history):
        motion = self.motion_decoder(z,motion_history)
        return motion

    def forward(self, other_motion_history, self_motion_history):
        current_omh_pose = other_motion_history[:,-1:,:]
        current_smh_pose = other_motion_history[:, -1:, :]
        current_pose = torch.cat((current_omh_pose,current_smh_pose),dim=2)
        results_dict = self.encode(current_pose)
        z = self.reparameterize(results_dict["mean"], results_dict["log_var"])
        reconstructed_motion = self.decode(z,other_motion_history)
        return reconstructed_motion, results_dict





