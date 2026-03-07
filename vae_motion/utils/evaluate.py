from te_fid import compute_interaction_fid
import torch
from utils.utils import DEVICE,breakup_tensor
from tqdm import tqdm
from te_fid import compute_interaction_fid
import random
import copy
def scramble_side(batch, side=0):
    batch = copy.deepcopy(batch)

    idx = list(range(len(batch)))
    perm = idx[:]
    random.shuffle(perm)

    for i, j in zip(idx, perm):
        batch[i][side] = batch[j][side]

    # pairwise temporal alignment for TE-FID
    for i in range(len(batch)):
        A, B = batch[i]
        T = min(A.shape[0], B.shape[0])
        batch[i][0] = A[:T]
        batch[i][1] = B[:T]

    return batch
def evaluate(mesh_renderer,smpl_manager,batch_size, vae_model, dataset, sequence_length=None, oracle_force=False):
    fake,real = generate_batch(batch_size, vae_model, dataset, sequence_length=None, oracle_force=True)
    te_fid = compute_interaction_fid(fake,real)

    import copy
    fake_scrambled = scramble_side(fake)
    real_scrambled = scramble_side(real)

    scrambled_te_fid = compute_interaction_fid(fake_scrambled, real_scrambled)

    print("real fid " + str(te_fid) + " "  + "scrambled " +str(scrambled_te_fid))


    fake_joints, fake_verts, _ = smpl_manager.smpl_forward(
        breakup_tensor(fake_scrambled[-1][0])["body_pose"].unsqueeze(0).to(DEVICE),
        breakup_tensor(fake_scrambled[-1][0])["global_orient"].unsqueeze(0).to(DEVICE),
        breakup_tensor(fake_scrambled[-1][0])["transl"].unsqueeze(0).to(DEVICE),
    )
    real_joints, real_verts, _ = smpl_manager.smpl_forward(
        breakup_tensor(fake_scrambled[-1][1])["body_pose"].unsqueeze(0).to(DEVICE),
        breakup_tensor(fake_scrambled[-1][1])["global_orient"].unsqueeze(0).to(DEVICE),
        breakup_tensor(fake_scrambled[-1][1])["transl"].unsqueeze(0).to(DEVICE),
    )
    mesh_renderer.save_mesh_twin_render_gif(
        fake_verts[0].detach().cpu(),
        real_verts[0].detach().cpu(),
        smpl_manager.model.faces,
        f"gifs//{torch.randint(8000, (1,)).item()}_fake_scrambled.gif",
    )
    fake_joints, fake_verts, _ = smpl_manager.smpl_forward(
        breakup_tensor(real_scrambled[-1][0])["body_pose"].unsqueeze(0).to(DEVICE),
        breakup_tensor(real_scrambled[-1][0])["global_orient"].unsqueeze(0).to(DEVICE),
        breakup_tensor(real_scrambled[-1][0])["transl"].unsqueeze(0).to(DEVICE),
    )
    real_joints, real_verts, _ = smpl_manager.smpl_forward(
        breakup_tensor(real_scrambled[-1][1])["body_pose"].unsqueeze(0).to(DEVICE),
        breakup_tensor(real_scrambled[-1][1])["global_orient"].unsqueeze(0).to(DEVICE),
        breakup_tensor(real_scrambled[-1][1])["transl"].unsqueeze(0).to(DEVICE),
    )
    mesh_renderer.save_mesh_twin_render_gif(
        fake_verts[0].detach().cpu(),
        real_verts[0].detach().cpu(),
        smpl_manager.model.faces,
        f"gifs//{torch.randint(8000, (1,)).item()}_real_scrambled.gif",
    )
    fake_joints, fake_verts, _ = smpl_manager.smpl_forward(
        breakup_tensor(real[-1][0])["body_pose"].unsqueeze(0).to(DEVICE),
        breakup_tensor(real[-1][0])["global_orient"].unsqueeze(0).to(DEVICE),
        breakup_tensor(real[-1][0])["transl"].unsqueeze(0).to(DEVICE),
    )
    real_joints, real_verts, _ = smpl_manager.smpl_forward(
        breakup_tensor(real[-1][1])["body_pose"].unsqueeze(0).to(DEVICE),
        breakup_tensor(real[-1][1])["global_orient"].unsqueeze(0).to(DEVICE),
        breakup_tensor(real[-1][1])["transl"].unsqueeze(0).to(DEVICE),
    )
    mesh_renderer.save_mesh_twin_render_gif(
        fake_verts[0].detach().cpu(),
        real_verts[0].detach().cpu(),
        smpl_manager.model.faces,
        f"gifs//{torch.randint(8000, (1,)).item()}_real.gif",
    )
    fake_joints, fake_verts, _ = smpl_manager.smpl_forward(
        breakup_tensor(fake[-1][0])["body_pose"].unsqueeze(0).to(DEVICE),
        breakup_tensor(fake[-1][0])["global_orient"].unsqueeze(0).to(DEVICE),
        breakup_tensor(fake[-1][0])["transl"].unsqueeze(0).to(DEVICE),
    )
    real_joints, real_verts, _ = smpl_manager.smpl_forward(
        breakup_tensor(fake[-1][1])["body_pose"].unsqueeze(0).to(DEVICE),
        breakup_tensor(fake[-1][1])["global_orient"].unsqueeze(0).to(DEVICE),
        breakup_tensor(fake[-1][1])["transl"].unsqueeze(0).to(DEVICE),
    )
    mesh_renderer.save_mesh_twin_render_gif(
        fake_verts[0].detach().cpu(),
        real_verts[0].detach().cpu(),
        smpl_manager.model.faces,
        f"gifs//{torch.randint(8000, (1,)).item()}_real.gif",
    )

    return te_fid

def generate_batch(batch_size, vae_model, dataset, sequence_length=None, oracle_force=True):
    generated_batch = []
    real_batch = []
    for batch_idx in range(batch_size):
        p1_true, p2_true = dataset.load_random_full_sequence()
        p1_true = p1_true.to(DEVICE)
        p2_true = p2_true.to(DEVICE)

        s_length = sequence_length if sequence_length else p1_true.shape[0]
        A_frames, B_frames = [], []

        A_hist = p1_true[:1].unsqueeze(0)
        B_hist = p2_true[:1].unsqueeze(0)

        with torch.no_grad():
            for i in tqdm(range(s_length)):
                # RE-ENCODE at each step with updated history
                mu, logvar = vae_model.interaction_encoder(A_hist, B_hist)
                z = vae_model.reparameterize(mu, logvar)

                A_next_pred, B_next_pred = vae_model.motion_decoder(z, A_hist, B_hist)

                A_next_pred = A_next_pred.detach()
                B_next_pred = B_next_pred.detach()

                if oracle_force:
                    A_hist = torch.cat([A_hist[:, 1:], p1_true[i].unsqueeze(0).unsqueeze(0)], dim=1)
                    B_hist = torch.cat([B_hist[:, 1:], p2_true[i].unsqueeze(0).unsqueeze(0)], dim=1)
                else:
                    A_hist = torch.cat([A_hist[:, 1:], A_next_pred.unsqueeze(0)], dim=1)
                    B_hist = torch.cat([B_hist[:, 1:], B_next_pred.unsqueeze(0)], dim=1)

                A_frames.append(A_next_pred)
                B_frames.append(B_next_pred)

        # Stack frames
        A_stacked = torch.stack(A_frames).squeeze().cpu()  # [T, D]
        B_stacked = torch.stack(B_frames).squeeze().cpu()  # [T, D]

        generated_batch.append([A_stacked,B_stacked])
        real_batch.append([p1_true.cpu(),p2_true.cpu()])


    return  generated_batch,real_batch
