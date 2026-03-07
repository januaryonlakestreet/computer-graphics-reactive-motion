import torch
import torch.nn.functional as F

def fit_smpl_sequence(smpl_model, joint_sequence, lr=1e-2, iters=300, device=None):
    """
    Fit a sequence of joint positions (T x 24 x 3) to SMPL poses via differentiable IK.

    Args:
        smpl_model: preloaded SMPL instance (from smplx.SMPL)
        joint_sequence: Tensor [T, 24, 3], target joint positions
        lr: learning rate for Adam
        iters: optimization steps per frame
        device: torch.device, optional

    Returns:
        dict with per-frame parameters:
            global_orient: [T, 3]
            body_pose: [T, 69]
            betas: [T, 10]
            transl: [T, 3]
            vertices: list of per-frame [6890, 3] arrays
            joints: list of per-frame [24, 3] arrays
    """

    if device is None:
        device = next(smpl_model.parameters()).device
    joint_sequence = joint_sequence.to(device)

    T = joint_sequence.shape[0]
    results = {
        'global_orient': [],
        'body_pose': [],
        'betas': [],
        'transl': [],
        'vertices': [],
        'joints': []
    }

    for t in range(T):
        target_joints = joint_sequence[t:t+1] - joint_sequence[t:t+1, [0], :]

        global_orient = torch.zeros(1, 3, requires_grad=True, device=device)
        body_pose = torch.zeros(1, 63, requires_grad=True, device=device)
        betas = torch.zeros(1, 10, requires_grad=True, device=device)
        transl = torch.zeros(1, 3, requires_grad=True, device=device)

        optimizer = torch.optim.Adam([global_orient, body_pose, betas, transl], lr=lr)

        for _ in range(iters):
            optimizer.zero_grad()
            output = smpl_model.smpl_forward(
                global_orient=global_orient,
                body_pose=body_pose,
                betas=betas,
                transl=transl,B=
            )
            pred_joints = output.joints[:, :24, :] - output.joints[:, [0], :]
            loss = torch.nn.functional.mse_loss(pred_joints, target_joints)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            output = smpl_model(
                global_orient=global_orient,
                body_pose=body_pose,
                betas=betas,
                transl=transl
            )

        results['global_orient'].append(global_orient.detach().squeeze(0))
        results['body_pose'].append(body_pose.detach().squeeze(0))
        results['betas'].append(betas.detach().squeeze(0))
        results['transl'].append(transl.detach().squeeze(0))
        results['vertices'].append(output.vertices[0].detach().cpu())
        results['joints'].append(output.joints[0, :24, :].detach().cpu())

    results['global_orient'] = torch.stack(results['global_orient'])
    results['body_pose'] = torch.stack(results['body_pose'])
    results['betas'] = torch.stack(results['betas'])
    results['transl'] = torch.stack(results['transl'])
    return results
def fit_smpl_to_joints(smpl_model, target_joints, lr=1e-2, iters=500, device=None):
    """
    Differentiable IK solver to estimate SMPL pose/shape from joint positions.

    Args:
        smpl_model: preloaded SMPL instance (e.g., from smplx.SMPL)
        target_joints: Tensor [1, 24, 3], target joint positions
        lr: learning rate for Adam
        iters: optimization iterations
        device: torch.device, optional

    Returns:
        dict with optimized parameters and mesh output
    """

    if device is None:
        device = next(smpl_model.parameters()).device
    target_joints = target_joints.to(device)

    # Normalize by root joint
    target_joints = target_joints - target_joints[:, [0], :]

    # Initialize optimization parameters
    global_orient = torch.zeros(1, 3, requires_grad=True, device=device)
    body_pose = torch.zeros(1, 69, requires_grad=True, device=device)
    betas = torch.zeros(1, 10, requires_grad=True, device=device)
    transl = torch.zeros(1, 3, requires_grad=True, device=device)

    optimizer = torch.optim.Adam([global_orient, body_pose, betas, transl], lr=lr)

    for i in range(iters):
        optimizer.zero_grad()

        output = smpl_model(
            global_orient=global_orient,
            body_pose=body_pose,
            betas=betas,
            transl=transl
        )

        pred_joints = output.joints[:, :24, :]
        pred_joints = pred_joints - pred_joints[:, [0], :]

        loss = F.mse_loss(pred_joints, target_joints)
        loss.backward()
        optimizer.step()

    # Final forward pass
    with torch.no_grad():
        output = smpl_model(
            global_orient=global_orient,
            body_pose=body_pose,
            betas=betas,
            transl=transl
        )

    return {
        'global_orient': global_orient.detach(),
        'body_pose': body_pose.detach(),
        'betas': betas.detach(),
        'transl': transl.detach(),
        'vertices': output.vertices.detach(),
        'joints': output.joints.detach(),
        'faces': smpl_model.faces
    }