import torch
import torch.nn as nn
import torch.nn.functional as F

class PoseEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        return self.encoder(x)  # [B, H]

class TemporalMemory(nn.Module):
    def __init__(self, hidden_dim, memory_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_size = memory_size
        self.lstm = nn.LSTMCell(hidden_dim, hidden_dim)
        self.memory = []  # list of LSTM hidden states

        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.val_proj = nn.Linear(hidden_dim, hidden_dim)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)

    def reset(self, batch_size, device):
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, device=device)
        self.memory = []
        self.h = h
        self.c = c

    def step(self, pose_embed):
        # pose_embed: [B, H]
        self.h, self.c = self.lstm(pose_embed, (self.h, self.c))
        # Append to memory
        self.memory.append(self.h.detach())  # detaching to prevent backprop through memory
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)

    def attend(self, query_embed):
        if not self.memory:
            return torch.zeros_like(query_embed)

        memory_stack = torch.stack(self.memory, dim=1)  # [B, T, H]
        Q = self.query_proj(query_embed).unsqueeze(1)  # [B, 1, H]
        K = self.key_proj(memory_stack)               # [B, T, H]
        V = self.val_proj(memory_stack)               # [B, T, H]

        scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.hidden_dim ** 0.5)  # [B, 1, T]
        attn_weights = F.softmax(scores, dim=-1)                                  # [B, 1, T]
        attended = torch.matmul(attn_weights, V).squeeze(1)                      # [B, H]

        return attended  # [B, H]

class TemporalAttentionMotionSynthesizer(nn.Module):
    def __init__(self, pose_dim, hidden_dim, memory_size, output_dim):
        super().__init__()
        self.encoder = PoseEncoder(pose_dim, hidden_dim)
        self.temporal_memory = TemporalMemory(hidden_dim, memory_size)
        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.predictor = nn.Linear(hidden_dim, output_dim)

    def reset_memory(self, batch_size, device):
        self.temporal_memory.reset(batch_size, device)

    def forward_step(self, pose):

        if pose.shape[0] is 1:
            print("f")
        """
        pose: [B, D] input pose at time t
        returns: [B, D_out] predicted next pose or representation
        """
        pose_embed = self.encoder(pose)                          # [B, H]
        context = self.temporal_memory.attend(pose_embed)        # [B, H]
        fused = self.fusion(torch.cat([pose_embed, context], dim=-1))  # [B, H]
        output = self.predictor(fused)                           # [B, D_out]
        self.temporal_memory.step(pose_embed)
        return output