import os

import torch.nn.functional as F
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
os.sys.path.append(parent_dir)
import random

import torch

from get_interaction import Get_Interaction
from utils.SMPLManager import SMPLManager
from settings.settings import args

from utils.mesh_renderer import MeshRenderer
from model.models import DartVAE
import torch.optim as optim

from utils.render import *
from utils.utils import breakup_tensor




def collect_history(a,b,data):
    collected = []
    for indx in range(len(a)):
        collected.append(data[b[indx]:a[indx]])

    if len(collected) == 0:
        print("f")
    return torch.stack(collected)

def kl_loss(mu, logvar):
    """
    Computes the KL divergence loss for a VAE.

    Args:
        mu (torch.Tensor): Mean of the latent distribution. Shape: (batch_size, latent_dim)
        logvar (torch.Tensor): Log-variance of the latent distribution. Shape: (batch_size, latent_dim)

    Returns:
        torch.Tensor: The KL divergence loss.
    """
    # Using the closed-form solution for Gaussian KL divergence
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return kl / mu.size(0)  # Averaging over the batch size







def main():


    smpl_manager = SMPLManager("smpl/", gender="neutral", device='cuda')
    mesh_renderer = MeshRenderer()
    args.num_epochs = 50000
    args.mini_batch_size = 128
    args.initial_lr = 0.00001
    args.final_lr = 1e-7





    frame_size =69 * 2

    pose_vae = DartVAE(
            motion_input_size=frame_size,
            motion_output_size=frame_size,
            normalization=None,
            history_length=args.history_length
        ).to(args.device)

    for ep in range(1, args.num_epochs + 1):
        mocap_data = Get_Interaction(args).Get_Interaction()

        history_length = 5
        sampler =  [random.randint(history_length, len(mocap_data["side_a"]) - history_length) for _ in range(args.mini_batch_size)]
        sampler_history =  [val - history_length  for val in sampler]

        rollout_loss = torch.tensor(0.0).to(args.device)
        pose_vae.optim.zero_grad()

        batch_raw_a = mocap_data["side_a"]
        batch_a = batch_raw_a[sampler].unsqueeze(dim=1)
        target_a = mocap_data["side_b"][sampler].unsqueeze(dim=1)
        batch_history_a = collect_history(sampler, sampler_history, batch_raw_a)

        batch_raw_b = mocap_data["side_b"]
        batch_b = batch_raw_b[sampler].unsqueeze(dim=1)
        target_b = mocap_data["side_a"][sampler].unsqueeze(dim=1)
        batch_history_b = collect_history(sampler, sampler_history, batch_raw_b)

        reconstructed_motion, results_dict = pose_vae(batch_history_a)



        recon_loss = F.mse_loss(batch_a,reconstructed_motion)

        kl = kl_loss(results_dict["mean"], results_dict["log_var"])
        rollout_loss += recon_loss + kl

        rollout_loss.backward()
        pose_vae.optim.step()

        if ep > 5 and ep % 100 == 0:
            print(f"{ep}    {rollout_loss.item():.6f}")


            mocap_data =  Get_Interaction(args).Get_Interaction()
            real_side = mocap_data["side_a"]
            sequence_length = real_side.shape[0]

            generated_side_b = mocap_data["side_a"][:history_length,:]

            for frame in range(history_length,sequence_length):
                batch = real_side[frame, :].unsqueeze(0).unsqueeze(0)
                history = mocap_data["side_a"][frame-history_length:frame,:].unsqueeze(0)
                new_side_b, _ = pose_vae(history)
                generated_side_b = torch.vstack((generated_side_b,new_side_b.squeeze().unsqueeze(0)))

            side_a = Get_Interaction(args).denormalize_a(real_side)
            side_b = Get_Interaction(args).denormalize_b(generated_side_b)

            fake_joints, fake_verts, _ = smpl_manager.smpl_forward(
                breakup_tensor(side_a)["body_pose"].unsqueeze(0),
                breakup_tensor(side_a)["global_orient"].unsqueeze(0),
                breakup_tensor(side_a)["transl"].unsqueeze(0),
            )
            real_joints, real_verts, _ = smpl_manager.smpl_forward(
                breakup_tensor(side_b)["body_pose"].unsqueeze(0),
                breakup_tensor(side_b)["global_orient"].unsqueeze(0),
                breakup_tensor(side_b)["transl"].unsqueeze(0),
            )
            mesh_renderer.save_mesh_twin_render_gif(
                fake_verts[0].detach().cpu(),
                real_verts[0].detach().cpu(),
                smpl_manager.model.faces,
                f"gifs//{torch.randint(8000, (1,)).item()}.gif",
            )



if __name__ == "__main__":
    main()
