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

    frame_size =69 * 2

    pose_vae = DartVAE(
            motion_input_size=frame_size,
            motion_output_size=frame_size,
            normalization=None,
            history_length=args.history_length
        ).to(args.device)

    for ep in range(1, args.num_epochs + 1):
        pose_vae.train()
        mocap_data = Get_Interaction(args).Get_Interaction()

        history_length = 5
        sampler =  [random.randint(history_length, len(mocap_data["side_a"]) - history_length) for _ in range(args.mini_batch_size)]
        sampler_history =  [val - history_length  for val in sampler]

        rollout_loss = torch.tensor(0.0).to(args.device)
        pose_vae.optim.zero_grad()

        batch_raw_a = mocap_data["side_a"]
        target_a = mocap_data["side_b"][sampler].unsqueeze(dim=1)
        batch_history_a = collect_history(sampler, sampler_history, batch_raw_a)

        batch_raw_b = mocap_data["side_b"]
        target_b = mocap_data["side_a"][sampler].unsqueeze(dim=1)
        batch_history_b = collect_history(sampler, sampler_history, batch_raw_b)
        bs = batch_history_b.shape[0]

        role_a = torch.zeros(bs, dtype=torch.long).to(args.device)
        role_b = torch.ones(bs, dtype=torch.long).to(args.device)

        roles = torch.cat([role_a, role_b], dim=0)
        histories_self = torch.cat((batch_history_a,batch_history_b),dim=0)
        histories_other = torch.cat((batch_history_b, batch_history_a), dim=0)

        targets = torch.cat((target_a,target_b),dim=0)

        reconstructed_motion, results_dict = pose_vae(histories_self,histories_other,roles)



        recon_loss = F.mse_loss(targets,reconstructed_motion)

        kl = kl_loss(results_dict["mean"], results_dict["log_var"])
        rollout_loss += recon_loss + kl

        rollout_loss.backward()
        pose_vae.optim.step()

        if ep > 5 and ep % 100 == 0:
            print(f"{ep}    {rollout_loss.item():.6f}")
            pose_vae.eval()

            mocap_data =  Get_Interaction(args).Get_Interaction()
            real_side = mocap_data["side_a"]
            sequence_length = real_side.shape[0]

            generated_side_b = mocap_data["side_b"][:history_length,:]
            generated_side_b = pose_vae.start_tokens(torch.ones(1).int().to('cuda')).squeeze()

            for frame in range(history_length, sequence_length):

                history = mocap_data["side_a"][frame - history_length:frame, :].unsqueeze(0)

                if generated_side_b.shape[0] < history_length:
                    pad = mocap_data["side_b"][:history_length - generated_side_b.shape[0]]
                    history_self = torch.vstack((pad, generated_side_b))[-history_length:]
                else:
                    history_self = generated_side_b[-history_length:]

                history_self = history_self.unsqueeze(0)

                new_side_b, _ = pose_vae(
                    history,
                    history_self,
                    torch.ones(1).int().to('cuda')
                )

                new_frame = new_side_b.squeeze().unsqueeze(0)
                generated_side_b = torch.vstack((generated_side_b, new_frame))

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
