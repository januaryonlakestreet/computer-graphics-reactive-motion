
import torch
import torch.nn.functional as F

def axis_angle_to_matrix(axis_angle):
    """
    axis_angle: [..., 3]
    returns: [..., 3, 3]
    """
    theta = torch.norm(axis_angle, dim=-1, keepdim=True)
    axis = axis_angle / (theta + 1e-8)

    x, y, z = axis.unbind(-1)

    zeros = torch.zeros_like(x)
    K = torch.stack([
        zeros, -z, y,
        z, zeros, -x,
        -y, x, zeros
    ], dim=-1).reshape(*axis.shape[:-1], 3, 3)

    I = torch.eye(3, device=axis_angle.device).expand_as(K)

    theta = theta.unsqueeze(-1)
    return I + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)


def matrix_to_axis_angle(R):
    """
    R: [..., 3, 3]
    returns: [..., 3]
    """
    cos_theta = (R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1 + 1e-6, 1 - 1e-6)
    theta = torch.acos(cos_theta)

    rx = R[..., 2, 1] - R[..., 1, 2]
    ry = R[..., 0, 2] - R[..., 2, 0]
    rz = R[..., 1, 0] - R[..., 0, 1]

    axis = torch.stack([rx, ry, rz], dim=-1)
    axis = F.normalize(axis, dim=-1)

    return axis * theta.unsqueeze(-1)


def matrix_to_6d(R):
    """
    R: [..., 3, 3]
    returns: [..., 6]
    """
    return R[..., :2].reshape(*R.shape[:-2], 6)


def rot6d_to_matrix(x):
    """
    x: [..., 6]
    returns: [..., 3, 3]
    """
    a1 = x[..., 0:3]
    a2 = x[..., 3:6]

    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)

    return torch.stack([b1, b2, b3], dim=-1)

def smpl_to_model_repr(batch_smpl):
    """
    batch_smpl: [B, T, 69]  (3 trans + 3 root + 63 body)
    returns: [B, T, 3 + 6 + 126]
    """

    B, T, D = batch_smpl.shape

    trans = batch_smpl[..., 0:3]
    root = batch_smpl[..., 3:6]
    body = batch_smpl[..., 6:]

    body = body.view(B, T, -1, 3)

    # Convert to rotation matrices
    root_R = axis_angle_to_matrix(root)
    body_R = axis_angle_to_matrix(body)

    # Convert to 6D
    root_6d = matrix_to_6d(root_R)
    body_6d = matrix_to_6d(body_R).reshape(B, T, -1)

    return torch.cat([trans, root_6d, body_6d], dim=-1)

def model_repr_to_smpl(batch_repr):
    """
    batch_repr: [B, T, 135]
    returns: [B, T, 69]
    """

    B, T, D = batch_repr.shape

    trans = batch_repr[..., 0:3]
    root_6d = batch_repr[..., 3:9]
    body_6d = batch_repr[..., 9:].view(B, T, -1, 6)

    root_R = rot6d_to_matrix(root_6d)
    body_R = rot6d_to_matrix(body_6d)

    root_aa = matrix_to_axis_angle(root_R)
    body_aa = matrix_to_axis_angle(body_R).reshape(B, T, -1)

    return torch.cat([trans, root_aa, body_aa], dim=-1)