import os

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
os.sys.path.append(parent_dir)
import torch.nn.functional as F
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
        self.role_embedding = nn.Embedding(2, hidden_dim)
        self.role_gate_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, motion_primitive,role_id):

        bs = motion_primitive.shape[0]
        motion_src = self.motion_proj(motion_primitive)
        role_token = self.role_embedding(role_id)  # (bs, hidden_dim)
        role_gate = torch.sigmoid(self.role_gate_proj(role_token))  # (bs, hidden_dim)

        dist = torch.tile(self.global_motion_token[:, None, :], (1, bs, 1))
        dist = dist.permute(1, 0, 2)
        dist = (dist * role_gate.unsqueeze(1) + role_token.unsqueeze(1))

        enc_vector = torch.cat((motion_src,dist),dim=1)

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
class Cross_Attention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, y, z):
        attn_out, _ = self.attn(x, y, z)
        return self.norm(x + attn_out)




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


class MambaBlock(nn.Module):
    """
    Pure-PyTorch Mamba-style SSM block.

    Implements the core selective state space mechanism without requiring
    the mamba_ssm CUDA package — suitable for prototyping. Swap this class
    out for `from mamba_ssm import Mamba` when moving to production.

    Architecture per block:
      x -> input_proj (expand) -> split into z (gate) and u (ssm input)
        u -> conv1d -> silu -> SSM(A, B, C, dt) -> gate(z) -> output_proj

    The SSM parameters A, B, C, dt are all input-dependent (selective),
    which is Mamba's key distinction from prior SSMs like S4.
    """
    def __init__(self, d_model, d_state=32, d_conv=4, expand=2):
        super().__init__()
        self.d_model  = d_model
        self.d_state  = d_state
        self.d_inner  = d_model * expand

        # input projection — doubles width, split into SSM input and gate
        self.in_proj  = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # short depthwise conv before SSM (captures local context)
        self.conv1d   = nn.Conv1d(
            in_channels=self.d_inner, out_channels=self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=True
        )

        # selective SSM projections — B, C, dt are input-dependent
        self.x_proj   = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)  # B, C, dt
        self.dt_proj  = nn.Linear(1, self.d_inner, bias=True)

        # A is fixed log-decay initialised as in the Mamba paper
        A = torch.arange(1, d_state + 1, dtype=torch.float).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log    = nn.Parameter(torch.log(A))

        self.D        = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm     = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (bs, L, d_model)
        residual = x
        bs, L, _  = x.shape

        xz        = self.in_proj(x)                          # (bs, L, d_inner*2)
        u, z      = xz.chunk(2, dim=-1)                      # each (bs, L, d_inner)

        # depthwise conv — truncate to sequence length
        u_conv    = self.conv1d(u.transpose(1, 2))[:, :, :L].transpose(1, 2)
        u_conv    = F.silu(u_conv)                            # (bs, L, d_inner)

        # selective parameters — derived from input
        x_dbc     = self.x_proj(u_conv)                      # (bs, L, 2*d_state + 1)
        B, C, dt  = x_dbc.split([self.d_state, self.d_state, 1], dim=-1)
        dt        = F.softplus(self.dt_proj(dt))             # (bs, L, d_inner)

        A         = -torch.exp(self.A_log)                   # (d_inner, d_state)

        # discretise
        dA        = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        # (bs, L, d_inner, d_state)
        dB        = dt.unsqueeze(-1) * B.unsqueeze(2)

        # Vectorised parallel prefix scan — replaces the Python for-loop.
        # Each hidden state h_t = dA_t * h_{t-1} + dB_t * u_t is a first-order
        # linear recurrence. We solve it in O(log L) passes using the standard
        # associative scan identity:
        #   combine(a2,b2, a1,b1) -> (a2*a1, a2*b1 + b2)
        # where the recurrence is h_t = a_t * h_{t-1} + b_t.
        # At L=5 (history window) this gives the exact same result as the loop
        # but keeps all ops on GPU without Python overhead.
        a = dA                                               # (bs, L, d_inner, d_state)
        b = dB * u_conv.unsqueeze(-1)                       # (bs, L, d_inner, d_state)

        # up-sweep: build prefix products/sums at power-of-2 strides
        L_pad = 1
        while L_pad < L:
            L_pad *= 2
        # pad to power of 2 for clean sweep (no-op padding: a=1, b=0)
        pad = L_pad - L
        if pad > 0:
            a = F.pad(a, (0, 0, 0, 0, 0, pad), value=1.0)
            b = F.pad(b, (0, 0, 0, 0, 0, pad), value=0.0)

        stride = 1
        while stride < L_pad:
            a_prev = a[:, :L_pad - stride]
            b_prev = b[:, :L_pad - stride]
            a_next = a[:, stride:]
            b_next = b[:, stride:]
            a = torch.cat([a[:, :stride],
                           a_next * a_prev], dim=1)
            b = torch.cat([b[:, :stride],
                           a_next * b_prev + b_next], dim=1)
            stride *= 2

        # b now holds cumulative hidden states h_0..h_{L-1} (zero initial state)
        h_seq = b[:, :L]                                    # (bs, L, d_inner, d_state)
        y     = (h_seq * C.unsqueeze(2)).sum(-1)            # (bs, L, d_inner)

        y         = y + u_conv * self.D.unsqueeze(0).unsqueeze(0)
        y         = y * F.silu(z)                            # gating

        out       = self.out_proj(y)                         # (bs, L, d_model)
        return self.norm(out + residual)

    def step(self, x_t, state):
        """
        Single-step recurrent update for O(1) webcam inference.
        x_t   : (bs, d_model)
        state : dict with keys 'h' (SSM hidden), 'conv_buf' (conv history)
                or None to initialise
        """
        bs = x_t.shape[0]

        if state is None:
            state = {
                "h":        torch.zeros(bs, self.d_inner, self.d_state, device=x_t.device),
                "conv_buf": torch.zeros(bs, self.d_inner, self.conv1d.kernel_size[0] - 1,
                                        device=x_t.device)
            }

        xz    = self.in_proj(x_t)                            # (bs, d_inner*2)
        u, z  = xz.chunk(2, dim=-1)

        # manual conv step — weight layout is (channels, 1, k), squeeze to (channels, k)
        conv_in  = torch.cat([state["conv_buf"], u.unsqueeze(-1)], dim=-1)
        weight   = self.conv1d.weight.squeeze(1)             # (d_inner, k)
        u_conv   = torch.einsum("bck,ck->bc", conv_in, weight) + self.conv1d.bias
        u_conv   = F.silu(u_conv)

        new_conv_buf = conv_in[:, :, 1:]                     # shift buffer

        x_dbc    = self.x_proj(u_conv)
        B, C, dt = x_dbc.split([self.d_state, self.d_state, 1], dim=-1)
        dt       = F.softplus(self.dt_proj(dt))

        A        = -torch.exp(self.A_log)
        dA       = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))
        dB       = dt.unsqueeze(-1) * B.unsqueeze(1)

        new_h    = dA * state["h"] + dB * u_conv.unsqueeze(-1)
        y        = (new_h * C.unsqueeze(1)).sum(-1)
        y        = y + u_conv * self.D
        y        = y * F.silu(z)

        out      = self.out_proj(y)                          # (bs, d_model)
        new_state = {"h": new_h, "conv_buf": new_conv_buf}
        return self.norm(out + x_t), new_state

class MambaHistoryEncoder(nn.Module):
    """
    Compresses a history window into a sequence of context vectors using
    stacked MambaBlocks. At training time the full window is processed in
    parallel. At inference (webcam mode) the state can be updated
    incrementally — O(1) per frame rather than re-attending over the full
    window each step.

    Output shape matches the input to cross-attention in the decoder:
      (bs, history_len, hidden_dim)
    """
    def __init__(self, input_dim, hidden_dim, num_layers=3, d_state=32, d_conv=4, expand=2):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.mamba_layers = nn.ModuleList([
            MambaBlock(d_model=hidden_dim, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(num_layers)
        ])

    def forward(self, history_frames):
        # history_frames: (bs, H, input_dim)
        x = self.input_proj(history_frames)       # (bs, H, hidden_dim)
        for layer in self.mamba_layers:
            x = layer(x)                          # (bs, H, hidden_dim)
        return x                                  # full sequence for cross-attn

    def step(self, frame, state):
        """
        Single-step incremental update for webcam inference.
        frame : (bs, 1, input_dim)
        state : list of per-layer state dicts, or None for initialisation
        Returns updated output token (bs, 1, hidden_dim) and new state list.
        """
        x = self.input_proj(frame).squeeze(1)     # (bs, hidden_dim)
        new_state = []
        for i, layer in enumerate(self.mamba_layers):
            s = state[i] if state is not None else None
            x, s_out = layer.step(x, s)
            new_state.append(s_out)
        return x.unsqueeze(1), new_state           # (bs, 1, hidden_dim), list[dict]




class SoftMoEFFN(nn.Module):
    """
    Role-aware Soft Mixture-of-Experts FFN.

    Expert layout (4 total):
      0, 1  — shared experts (interaction-level patterns, role-agnostic)
      2     — agent A expert  (role 0 specific)
      3     — agent B expert  (role 1 specific)

    The router produces weights over all 4 experts. A learned per-role bias
    tilts routing toward the agent-specific expert without hard-assigning it,
    so shared experts remain accessible to both roles when useful.
    """
    def __init__(self, dim, ff_dim, dropout, num_shared=2, num_per_agent=1):
        super().__init__()
        total_experts      = num_shared + num_per_agent * 2
        self.total_experts = total_experts
        self.router        = nn.Linear(dim, total_experts)
        self.role_bias     = nn.Embedding(2, total_experts)   # learned bias per role
        self.experts       = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, ff_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ff_dim, dim),
            )
            for _ in range(total_experts)
        ])

    def forward(self, x, role_id):
        # x:       (bs, seq, dim)
        # role_id: (bs,)
        logits  = self.router(x)                               # (bs, seq, total_experts)
        bias    = self.role_bias(role_id)                      # (bs, total_experts)
        weights = F.softmax(logits + bias.unsqueeze(1), dim=-1)

        expert_outs = torch.stack(
            [expert(x) for expert in self.experts], dim=-1    # (bs, seq, dim, total_experts)
        )
        return (expert_outs * weights.unsqueeze(-2)).sum(dim=-1)  # (bs, seq, dim)


class DecoderTransformerLayer(nn.Module):
    def __init__(self, latent_dim, num_heads=4, dropout=0.1, ff_dim=1024,
                 num_shared=2, num_per_agent=1):
        super().__init__()
        self.self_attn = Self_Attention(latent_dim, num_heads, dropout)

        # separate cross-attn streams: own history, other side's history, text prompt
        self.cross_attn_own   = Cross_Attention(latent_dim, num_heads, dropout)
        self.cross_attn_other = Cross_Attention(latent_dim, num_heads, dropout)


        # independent role-aware Soft-MoE FFN at every position
        self.moe_self  = SoftMoEFFN(latent_dim, ff_dim, dropout, num_shared, num_per_agent)
        self.moe_own   = SoftMoEFFN(latent_dim, ff_dim, dropout, num_shared, num_per_agent)
        self.moe_other = SoftMoEFFN(latent_dim, ff_dim, dropout, num_shared, num_per_agent)
        self.moe_text  = SoftMoEFFN(latent_dim, ff_dim, dropout, num_shared, num_per_agent)

        self.final_out = nn.Linear(latent_dim, latent_dim)
        self.layernorm = nn.LayerNorm(latent_dim)

    def forward(self, transformer_in, own_memory, other_memory, role_id):
        # self-attention + role-aware MoE
        t_in = self.self_attn(transformer_in, transformer_in, transformer_in)
        transformer_in = self.layernorm(t_in + transformer_in)
        transformer_in = self.moe_self(transformer_in, role_id)

        # cross-attend to own history — "what was I doing?"
        t_in = self.cross_attn_own(transformer_in, own_memory, own_memory)
        transformer_in = self.layernorm(t_in + transformer_in)
        transformer_in = self.moe_own(transformer_in, role_id)

        # cross-attend to other side's history — "what were they doing?"
        t_in = self.cross_attn_other(transformer_in, other_memory, other_memory)
        transformer_in = self.layernorm(t_in + transformer_in)
        transformer_in = self.moe_other(transformer_in, role_id)



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
        num_mamba_layers = 3
        mamba_d_state = 8
        self.history_encoder = MambaHistoryEncoder(69, hidden_dim,
                                                       num_mamba_layers, mamba_d_state)


    def forward(self, z, other_history_frames,self_history_frames,role_id):
        self.history_length = other_history_frames.shape[1]
        # 1. Base latent processing
        memory = self.latent_proj(z).unsqueeze(1)

        # 3. History embedding (specialized)
        other_history_embed = self.history_encoder(other_history_frames)
        other_history_embed = self.pos_encoder(other_history_embed)

        self_history_embed = self.history_encoder(self_history_frames)
        self_history_embed = self.pos_encoder(self_history_embed)

        memory = self.pos_encoder(memory)

        # 6. Transformer processing
        layer_in = memory
        for layer in self.new_transformer_decoder:
            layer_in = layer(layer_in, other_history_embed,self_history_embed,role_id)

        # 7. Final projection
        return self.output_proj(layer_in)


class InteractionStartTokens(nn.Module):
    """
    Learned per-role starting pose embeddings, projected from text.

    Produces a plausible text-consistent initial pose for each character
    before any real history exists. Role asymmetry is enforced by construction
    — role 0 and role 1 have separate base embeddings so they learn to occupy
    distinct spatial configurations relative to each other.

    Training signal: reconstruction loss on the first decoded frame
    backpropagates naturally through the decoder into these tokens.
    No separate loss is needed.

    Usage:
        start_a = start_tokens(role_a, text_emb)   # (bs, frame_size)
        start_b = start_tokens(role_b, text_emb)   # (bs, frame_size)
    """
    def __init__(self, hidden_dim, frame_size,history_length):
        super().__init__()
        frame_size = int(frame_size//2)
        # separate base token per role — the core source of spatial asymmetry
        self.base_token = nn.Embedding(2, hidden_dim)
        # text shifts the starting config toward the interaction type
        self.text_proj  = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # project to pose space
        self.to_pose  = nn.Linear(hidden_dim, frame_size*history_length)

    def forward(self, role_id, text_emb=None):
        # role_id : (bs,)   text_emb : (bs, hidden_dim)
        base = self.base_token(role_id)             # (bs, hidden_dim)
        #cond = F.silu(self.text_proj(text_emb))     # (bs, hidden_dim)
        to_frame_size =  self.to_pose(base) # + cond           # (bs, frame_size)
        return to_frame_size.reshape(1,5,69)



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

        # learned per-role starting poses — text-conditioned, role-asymmetric
        self.start_tokens = InteractionStartTokens(
            hidden_dim=hidden_dim,
            frame_size=motion_output_size,
            history_length=5
        ).to(self.device)

        self.latent_dim = latent_dim
        self.optim = optim.Adam(list(self.motion_encoder.parameters())+list(self.motion_decoder.parameters()), lr=3e-4)


    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, other_motion_history, self_motion_history,role_id):
        current_omh_pose = other_motion_history[:,-1:,:]
        current_smh_pose = self_motion_history[:, -1:, :]
        current_pose =torch.cat((current_omh_pose,current_smh_pose),dim=2)
        mean, log_var = self.motion_encoder(current_pose,role_id)
        results_dict =  {
                "mean":mean,
                "log_var":log_var,
                }
        z = self.reparameterize(results_dict["mean"], results_dict["log_var"])

        reconstructed_motion = self.motion_decoder(z,other_motion_history,self_motion_history,role_id)


        return reconstructed_motion,results_dict





