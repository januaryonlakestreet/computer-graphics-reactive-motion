from types import SimpleNamespace
import torch
import os

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
os.sys.path.append(parent_dir)
env_path = os.path.join(parent_dir, "environments")
args = SimpleNamespace(
    device="cuda:0" if torch.cuda.is_available() else "cpu",
    mocap_file=os.path.join(env_path, "mocap.npz"),
    norm_mode="zscore",
    latent_size=128,
    num_embeddings=12,
    num_experts=6,
    num_condition_frames=1,
    num_future_predictions=1,
    sequence_length = 5,
    min_sequence_length=5,
    num_steps_per_rollout=5,
    future_length=1,
    history_length=1,
    kl_beta=1e-6,
    load_saved_model=True,
)


