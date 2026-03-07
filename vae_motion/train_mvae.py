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
from torch.cuda.amp import GradScaler


def interaction_contact_labels(positions_a,positions_b,thres= 0.00001):
    return torch.from_numpy((positions_a-positions_b < thres).astype(np.float32))


def generate_contact_labels(side_a,side_b):
    return interaction_contact_labels(np.array(side_a[:,:,3:69].detach().cpu()),np.array(side_b[:,:,3:69].detach().cpu())).to('cuda')


def foot_detect(positions, thres=0.1):
    fid_l = [7, 10]
    fid_r = [8, 11]
    velfactor, heightfactor = (torch.from_numpy(np.array([thres, thres])).to('cuda'),
                               torch.from_numpy(np.array([0.1, 0.05])).to('cuda'))

    feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
    feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
    feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
    feet_l_h = positions[:-1, fid_l, 1]
    feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).float()

    feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
    feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
    feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
    feet_r_h = positions[:-1, fid_r, 1]
    feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).float()
    return torch.cat((torch.zeros(1,2).to('cuda'),feet_l)), torch.cat((torch.zeros(1,2).to('cuda'),feet_r))

def CalculateFootLoss(real_side_b,fake_side_b):

    real_fc_labels = real_side_b[:,:,-70:-66].squeeze()
    fake_joints = fake_side_b[:,:,3:69].squeeze().reshape(-1,22,3)

    fake_fc_labels_l,fake_fc_labels_r = foot_detect(fake_joints, 0.00001)
    fake_fc_labels = torch.concatenate((fake_fc_labels_l,fake_fc_labels_l),dim=1)

    fc_loss = torch.nn.MSELoss()(real_fc_labels,fake_fc_labels)

    return fc_loss



def CalculateInteractionLabelLoss(side_a,real_b,fake_b):
    side_a_i_labels = side_a[:,:,-66:]
    real_b_i_labels = real_b[:,:,-66:]

    fake_b_i_labels = generate_contact_labels(side_a,fake_b)

    loss = torch.nn.MSELoss()(real_b_i_labels,fake_b_i_labels)
    return loss


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

        batch_raw = mocap_data["side_a"]
        batch = batch_raw[sampler].unsqueeze(dim=1)
        target = mocap_data["side_b"][sampler].unsqueeze(dim=1)
        batch_history = collect_history(sampler, sampler_history, batch_raw)

        reconstructed_motion, results_dict = pose_vae(batch, batch_history)



        recon_loss = F.mse_loss(target,reconstructed_motion)

        kl = kl_loss(results_dict["mean"], results_dict["log_var"])
        rollout_loss += recon_loss + kl

        rollout_loss.backward()
        pose_vae.optim.step()

        if ep > 5 and ep % 100 == 0:
            print(f"{ep}    {rollout_loss.item():.6f}")
            device = 'cuda'

            mocap_data =  Get_Interaction(args).Get_Interaction()
            real_side = mocap_data["side_a"]
            sequence_length = real_side.shape[0]

            generated_side_b = mocap_data["side_b"][:history_length,:]

            for frame in range(history_length,sequence_length):
                batch = real_side[frame, :].unsqueeze(0).unsqueeze(0)
                history = real_side[frame-history_length:frame,:].unsqueeze(0)
                new_side_b, _ = pose_vae(batch, history)
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
