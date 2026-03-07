import os
import torch
import torch.nn.functional as F
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
os.sys.path.append(parent_dir)
import random
import numpy as np
import torch
import copy
from get_interaction import Get_Interaction
from utils.SMPLManager import SMPLManager
from settings.settings import args
from utils.maths import quaternion_to_axis_angle,batched_quaternion_to_axis_angle
from utils.mesh_renderer import MeshRenderer
from model.models import (
DartVAE,MotionDiscriminator
)
import torch.optim as optim
from utils.matplotvis import generate_rollout
from utils.render import *
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

    args.num_epochs = 50000
    args.mini_batch_size = 128
    args.initial_lr = 0.00001
    args.final_lr = 1e-7




    raw_data_1 = np.load("../interaction_dataset/processed/5883_1.npy")
    raw_data_2 = np.load("../interaction_dataset/processed/5883_2.npy")
    mocap_data_1 = torch.from_numpy(raw_data_1).float().to(args.device)
    mocap_data_2 = torch.from_numpy(raw_data_2).float().to(args.device)



    loaded_stats_a = np.load("../interaction_dataset/processed/stats/stats_a.npy", allow_pickle=True).item()
    loaded_stats_b = np.load("../interaction_dataset/processed/stats/stats_b.npy", allow_pickle=True).item()

    avg_a = torch.from_numpy(loaded_stats_a['mean']).float().to(args.device)
    std_a = torch.from_numpy(loaded_stats_a['std']).float().to(args.device)
    min_a = torch.from_numpy(loaded_stats_a['min']).float().to(args.device)
    max_a = torch.from_numpy(loaded_stats_a['max']).float().to(args.device)

    avg_b = torch.from_numpy(loaded_stats_b['mean']).float().to(args.device)
    std_b = torch.from_numpy(loaded_stats_b['std']).float().to(args.device)
    min_b = torch.from_numpy(loaded_stats_b['min']).float().to(args.device)
    max_b = torch.from_numpy(loaded_stats_b['max']).float().to(args.device)
    std_a[std_a == 0] = 1.0
    std_b[std_b == 0] = 1.0
    avg = torch.cat((avg_a, avg_b))
    std = torch.cat((std_a, std_b))
    min = torch.cat((min_a, min_b))
    max = torch.cat((max_a, max_b))
    # Make sure we don't divide by 0
    std[std == 0] = 1.0


    normalization = {
        "mode": args.norm_mode,
        "max": max,
        "min": min,
        "avg": avg,
        "std": std,
    }
    frame_size = mocap_data_1.size()[1] * 2

    pose_vae = DartVAE(
            motion_input_size=frame_size,
            motion_output_size=frame_size,
            normalization=normalization,
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

        real = pose_vae.denormalize(torch.cat((batch, target), dim=2))
        fake = pose_vae.denormalize(torch.cat((batch, reconstructed_motion), dim=2))

        side_a = real[:, :, :293]
        real_b = real[:, :, 293:]
        fake_b = fake[:, :, 293:]


        interaction_label_loss = CalculateInteractionLabelLoss(side_a,real_b,fake_b)
        fc_loss = CalculateFootLoss(real_b,fake_b)

        recon_loss = torch.nn.MSELoss()(real_b, fake_b)

        kl = kl_loss(results_dict["mean"], results_dict["log_var"])
        rollout_loss += recon_loss + kl + interaction_label_loss + fc_loss

        rollout_loss.backward()
        pose_vae.optim.step()

        if ep > 5 and ep % 100 == 0:
            print(f"{ep}    {rollout_loss.item():.6f}")
            device = 'cuda'
            real_frames, fake_frames = generate_rollout(pose_vae, Get_Interaction, args)

if __name__ == "__main__":
    main()
