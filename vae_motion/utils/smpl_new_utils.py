"""
Derivative-first SMPL(-X core) utility for stable autoregressive training.

Key changes:
- Velocity / angular-velocity primary representation
- Joint angular velocity in so(3) tangent space
- Acceleration features included
- Optional shared pair canonicalization
- Dataset-scale normalization hooks
- Absolute reconstruction via cumulative integration
"""

from copy import deepcopy
from typing import Tuple, Dict, Optional

import torch
import torch.nn.functional as F
from pytorch3d import transforms


# ─────────────────────────────────────────────────────────────
#  Lie algebra helpers
# ─────────────────────────────────────────────────────────────

def so3_log(R: torch.Tensor) -> torch.Tensor:
    """Rotation matrix [...,3,3] → axis-angle (tangent) [...,3]."""
    return transforms.matrix_to_axis_angle(R)


def so3_exp(w: torch.Tensor) -> torch.Tensor:
    """Axis-angle [...,3] → rotation matrix [...,3,3]."""
    return transforms.axis_angle_to_matrix(w)


def relative_rot(R_t: torch.Tensor, R_prev: torch.Tensor) -> torch.Tensor:
    """R_rel = R_t @ R_prev^T."""
    return R_t @ R_prev.transpose(-1, -2)


# ─────────────────────────────────────────────────────────────
#  Core Utility
# ─────────────────────────────────────────────────────────────

class BodyUtility:

    """
    Derivative representation:

    Frame 0:
        transl_0              [B,1,3]
        global_orient_0       [B,1,3,3]
        body_pose_0           [B,1,21,3,3]

    For t >= 1:
        transl_vel            [B,T-1,3]
        global_ang_vel        [B,T-1,3]
        body_ang_vel          [B,T-1,21,3]

    Optional accelerations computed on demand.
    """

    # ─────────────────────────────────────────────────────────
    #  Canonicalization
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def canonicalize_pair(
        seq_a: Dict[str, torch.Tensor],
        seq_b: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict, Dict]:
        """
        Shared interaction frame:

        origin = midpoint pelvis at t0
        x-axis = A→B projected to floor
        z-axis = world up
        """

        seq_a = deepcopy(seq_a)
        seq_b = deepcopy(seq_b)

        B = seq_a["transl"].shape[0]
        device = seq_a["transl"].device

        pelvis_a = seq_a["transl"][:, 0]
        pelvis_b = seq_b["transl"][:, 0]

        origin = ((pelvis_a + pelvis_b) / 2.0).unsqueeze(1)

        x_axis = pelvis_b - pelvis_a
        x_axis[:, 2] = 0
        x_axis = F.normalize(x_axis, dim=-1)

        z_axis = torch.zeros_like(x_axis)
        z_axis[:, 2] = 1.0

        y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=-1), dim=-1)

        R = torch.stack([x_axis, y_axis, z_axis], dim=-1)
        R_inv = R.transpose(-1, -2)

        for seq in (seq_a, seq_b):
            seq["transl"] = (seq["transl"] - origin) @ R_inv
            seq["global_orient"] = R_inv.unsqueeze(1) @ seq["global_orient"]

        return R, origin, seq_a, seq_b


    # ─────────────────────────────────────────────────────────
    #  Derivative Feature Extraction
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def compute_derivatives(seq: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        B, T, _ = seq["transl"].shape

        # transl velocity
        transl_vel = seq["transl"][:, 1:] - seq["transl"][:, :-1]

        # global angular velocity (Lie algebra)
        go = seq["global_orient"]
        go_rel = relative_rot(go[:, 1:], go[:, :-1])
        global_ang_vel = so3_log(go_rel)

        # body angular velocity per joint
        bp = seq["body_pose"]
        bp_rel = relative_rot(bp[:, 1:], bp[:, :-1])
        body_ang_vel = so3_log(bp_rel)

        return {
            "transl_0": seq["transl"][:, :1],
            "global_orient_0": seq["global_orient"][:, :1],
            "body_pose_0": seq["body_pose"][:, :1],
            "transl_vel": transl_vel,
            "global_ang_vel": global_ang_vel,
            "body_ang_vel": body_ang_vel,
        }


    # ─────────────────────────────────────────────────────────
    #  Acceleration (for smoothness loss)
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def compute_acceleration(vel: torch.Tensor) -> torch.Tensor:
        return vel[:, 1:] - vel[:, :-1]


    # ─────────────────────────────────────────────────────────
    #  Reconstruction from derivatives
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def integrate_derivatives(features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        transl_0 = features["transl_0"]
        global_0 = features["global_orient_0"]
        body_0   = features["body_pose_0"]

        v = features["transl_vel"]
        w_g = features["global_ang_vel"]
        w_b = features["body_ang_vel"]

        B, Tm1, _ = v.shape
        T = Tm1 + 1

        transl = [transl_0]
        global_orient = [global_0]
        body_pose = [body_0]

        for t in range(Tm1):

            transl.append(transl[-1] + v[:, t:t+1])

            R_next = so3_exp(w_g[:, t:t+1]) @ global_orient[-1]
            global_orient.append(R_next)

            Rb_next = so3_exp(w_b[:, t:t+1]) @ body_pose[-1]
            body_pose.append(Rb_next)

        return {
            "transl": torch.cat(transl, dim=1),
            "global_orient": torch.cat(global_orient, dim=1),
            "body_pose": torch.cat(body_pose, dim=1),
        }


    # ─────────────────────────────────────────────────────────
    #  Normalization Hooks
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def normalize_derivatives(features: Dict[str, torch.Tensor],
                              stats: Dict[str, torch.Tensor]):

        out = deepcopy(features)

        out["transl_vel"] = out["transl_vel"] / stats["transl_vel_std"]
        out["global_ang_vel"] = out["global_ang_vel"] / stats["global_ang_vel_std"]
        out["body_ang_vel"] = out["body_ang_vel"] / stats["body_ang_vel_std"]

        return out


    @staticmethod
    def denormalize_derivatives(features: Dict[str, torch.Tensor],
                                stats: Dict[str, torch.Tensor]):

        out = deepcopy(features)

        out["transl_vel"] = out["transl_vel"] * stats["transl_vel_std"]
        out["global_ang_vel"] = out["global_ang_vel"] * stats["global_ang_vel_std"]
        out["body_ang_vel"] = out["body_ang_vel"] * stats["body_ang_vel_std"]

        return out

    @classmethod
    def encode_pair(
            cls,
            seq_a: Dict[str, torch.Tensor],
            seq_b: Dict[str, torch.Tensor],
            stats: Optional[Dict[str, torch.Tensor]] = None,
            device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode a pair of sequences into derivative feature space.

        Returns dict containing:
            - feature_vec        [B, T-1, D]
            - init_state_a
            - init_state_b
            - canonical_R
            - canonical_t
            - stats (if used)
        """

        seq_a = deepcopy(seq_a)
        seq_b = deepcopy(seq_b)

        B, T, _ = seq_a["transl"].shape
        device = device or seq_a["transl"].device

        # ── Axis-angle → rotation matrices ────────────────────
        for seq in (seq_a, seq_b):
            seq["global_orient"] = transforms.axis_angle_to_matrix(seq["global_orient"])
            seq["body_pose"] = transforms.axis_angle_to_matrix(
                seq["body_pose"].reshape(B, T, 21, 3)
            )

        # ── Shared canonicalization ───────────────────────────
        R_can, t_can, seq_a, seq_b = cls.canonicalize_pair(seq_a, seq_b)

        # ── Derivatives ───────────────────────────────────────
        feat_a = cls.compute_derivatives(seq_a)
        feat_b = cls.compute_derivatives(seq_b)

        # ── Optional normalization ────────────────────────────
        if stats is not None:
            feat_a = cls.normalize_derivatives(feat_a, stats)
            feat_b = cls.normalize_derivatives(feat_b, stats)

        # ── Flatten feature vectors (velocities only) ─────────
        def flatten(feat):
            return torch.cat([
                feat["transl_vel"],  # [B,T-1,3]
                feat["global_ang_vel"],  # [B,T-1,3]
                feat["body_ang_vel"].reshape(B, T - 1, -1),  # [B,T-1,63]
            ], dim=-1)

        vec_a = flatten(feat_a)
        vec_b = flatten(feat_b)

        feature_vec = torch.cat([vec_a, vec_b], dim=-1).to(device)
        # final dim = 69 * 2 = 138 per timestep

        return {
            "feature_vec": feature_vec,  # model input
            "init_state_a": {
                "transl_0": feat_a["transl_0"],
                "global_orient_0": feat_a["global_orient_0"],
                "body_pose_0": feat_a["body_pose_0"],
            },
            "init_state_b": {
                "transl_0": feat_b["transl_0"],
                "global_orient_0": feat_b["global_orient_0"],
                "body_pose_0": feat_b["body_pose_0"],
            },
            "canonical_R": R_can,
            "canonical_t": t_can,
            "stats": stats,
        }

    @classmethod
    def decode_pair(
            cls,
            feature_vec: torch.Tensor,  # [B, T-1, 138]
            reconstruction_state: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Reconstruct both sequences from feature vector + saved state.
        """

        B, Tm1, D = feature_vec.shape
        T = Tm1 + 1

        init_a = reconstruction_state["init_state_a"]
        init_b = reconstruction_state["init_state_b"]
        R_can = reconstruction_state["canonical_R"]
        t_can = reconstruction_state["canonical_t"]
        stats = reconstruction_state.get("stats", None)

        # split feature vector
        vec_a, vec_b = torch.chunk(feature_vec, 2, dim=-1)

        def unflatten(vec, init_state):
            transl_vel = vec[..., 0:3]
            global_ang_vel = vec[..., 3:6]
            body_ang_vel = vec[..., 6:].reshape(B, Tm1, 21, 3)

            feat = {
                **init_state,
                "transl_vel": transl_vel,
                "global_ang_vel": global_ang_vel,
                "body_ang_vel": body_ang_vel,
            }

            if stats is not None:
                feat = cls.denormalize_derivatives(feat, stats)

            return cls.integrate_derivatives(feat)

        seq_a = unflatten(vec_a, init_a)
        seq_b = unflatten(vec_b, init_b)

        # ── Transform back to world space ─────────────────────
        for seq in (seq_a, seq_b):
            seq["transl"] = seq["transl"] @ R_can + t_can
            seq["global_orient"] = R_can.unsqueeze(1) @ seq["global_orient"]

        # ── Convert back to axis-angle ────────────────────────
        def to_axis_angle(seq):
            return {
                "transl": seq["transl"],
                "global_orient": transforms.matrix_to_axis_angle(seq["global_orient"]),
                "body_pose": transforms.matrix_to_axis_angle(
                    seq["body_pose"].reshape(-1, 3, 3)
                ).reshape(B, T, 63),
            }

        return to_axis_angle(seq_a), to_axis_angle(seq_b)

