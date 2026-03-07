"""
Simplified body utility for SMPL-X sequences.

Expects body_param dicts with keys:
    transl        : [..., 3]
    global_orient : [..., 3, 3]   (rotation matrix)
    body_pose     : [..., 21, 3, 3]

No PCA hands, no betas, no face — just the core kinematic chain.
"""

from copy import deepcopy
from typing import Tuple, Dict, Optional

import torch
import torch.nn.functional as F
from pytorch3d import transforms


# ─────────────────────────────────────────────────────────────
#  Rotation helpers
# ─────────────────────────────────────────────────────────────

def aa_to_rotmat(x: torch.Tensor) -> torch.Tensor:
    """Axis-angle [..., 3] → rotation matrix [..., 3, 3]."""
    return transforms.axis_angle_to_matrix(x)


def rotmat_to_6d(x: torch.Tensor) -> torch.Tensor:
    """Rotation matrix [..., 3, 3] → 6-D rep [..., 6]."""
    return transforms.matrix_to_rotation_6d(x)


def sixd_to_rotmat(x: torch.Tensor) -> torch.Tensor:
    """6-D rep [..., 6] → rotation matrix [..., 3, 3]."""
    return transforms.rotation_6d_to_matrix(x)


def from_axis_angle(param: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Convert a raw SMPL param dict with axis-angle rotations into rotation matrices.
    Expects global_orient [..., 3] and body_pose [..., 63] (21 joints × 3).
    """
    param = deepcopy(param)
    prefix = param["global_orient"].shape[:-1]          # e.g. (B,) or (B, T)

    param["global_orient"] = aa_to_rotmat(
        param["global_orient"]
    )                                                    # [..., 3, 3]

    param["body_pose"] = aa_to_rotmat(
        param["body_pose"].reshape(*prefix, 21, 3)
    )                                                    # [..., 21, 3, 3]

    return param


# ─────────────────────────────────────────────────────────────
#  Coordinate-frame helpers
# ─────────────────────────────────────────────────────────────

def get_new_coordinate(
    pelvis: torch.Tensor,
    left_hip: torch.Tensor,
    right_hip: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build a body-centred coordinate frame from pelvis + hip joints
    (all in world space, shape [B, 3]).

    x-axis : right → left hip direction (projected to floor)
    z-axis : world up (0,0,1)
    y-axis : cross(z, x)

    Returns:
        rotmat  [B, 3, 3]   columns are [x, y, z] axes
        transl  [B, 1, 3]   pelvis origin
    """
    x_axis = right_hip - left_hip               # [B, 3]   (or left-right, just pick one)
    x_axis = x_axis.clone()
    x_axis[:, 2] = 0.0                          # project onto floor
    x_axis = F.normalize(x_axis, dim=-1)

    z_axis = torch.zeros_like(x_axis)
    z_axis[:, 2] = 1.0                          # world up

    y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=-1), dim=-1)

    rotmat = torch.stack([x_axis, y_axis, z_axis], dim=-1)   # [B, 3, 3]
    transl = pelvis.unsqueeze(1)                              # [B, 1, 3]
    return rotmat, transl


# ─────────────────────────────────────────────────────────────
#  Transform utilities
# ─────────────────────────────────────────────────────────────

def transform_points_to_local(
    points: torch.Tensor,       # [B, N, 3]
    rotmat: torch.Tensor,       # [B, 3, 3]  world → local
    origin: torch.Tensor,       # [B, 1, 3]
) -> torch.Tensor:
    """Bring world-space points into the local coordinate frame."""
    return torch.einsum("bij,bnj->bni", rotmat, points - origin)


def transform_points_to_world(
    points: torch.Tensor,       # [B, N, 3]
    rotmat: torch.Tensor,       # [B, 3, 3]  local → world
    origin: torch.Tensor,       # [B, 1, 3]
) -> torch.Tensor:
    """Bring local-frame points back to world space."""
    return torch.einsum("bij,bnj->bni", rotmat, points) + origin


def compose_transforms(
    R1: torch.Tensor, t1: torch.Tensor,   # parent  [B,3,3], [B,1,3]
    R2: torch.Tensor, t2: torch.Tensor,   # child   [B,3,3], [B,1,3]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Chain two rigid transforms:  T_total = T1 ∘ T2."""
    R = torch.einsum("bij,bjk->bik", R1, R2)
    t = torch.einsum("bij,bnj->bni", R1, t2) + t1
    return R, t


# ─────────────────────────────────────────────────────────────
#  Core utility class
# ─────────────────────────────────────────────────────────────

class BodyUtility:
    """
    Lightweight utility for SMPL-X sequences represented as:
        transl        : [B, T, 3]
        global_orient : [B, T, 3, 3]
        body_pose     : [B, T, 21, 3, 3]

    No body model required — all operations are pure geometry.
    If you need joint positions you'll still need to run SMPL-X,
    but all coordinate-frame canonicalization works without it.
    """

    # ── Feature packing / unpacking ─────────────────────────────────────────
    #
    #  Frame features  [B, T, D]  — one entry per timestep
    #  Delta features  [B, T-1, D] — one entry per consecutive pair
    #
    #  They have different time dimensions so they cannot share a single
    #  flat tensor.  Use dict_to_tensors() / tensors_to_dict() which
    #  returns / accepts a (frame_tensor, delta_tensor) pair.

    FRAME_REPR = {
        "transl":   3,
        "poses_6d": 22 * 6,   # global_orient(6) + body_pose(21×6)
    }
    DELTA_REPR = {
        "transl_delta":            3,
        "global_orient_delta_6d":  6,
    }
    FRAME_DIM = sum(FRAME_REPR.values())   # 3 + 132 = 135
    DELTA_DIM = sum(DELTA_REPR.values())   # 3 + 6   = 9

    # Keep a flat MOTION_REPR for code that only cares about frame features
    MOTION_REPR = FRAME_REPR
    FEATURE_DIM = FRAME_DIM

    @classmethod
    def dict_to_tensors(
        cls, d: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            frame_tensor  [B, T,   FRAME_DIM]
            delta_tensor  [B, T-1, DELTA_DIM]
        """
        frame = torch.cat([d[k] for k in cls.FRAME_REPR], dim=-1)
        delta = torch.cat([d[k] for k in cls.DELTA_REPR], dim=-1)
        return frame, delta

    @classmethod
    def tensors_to_dict(
        cls,
        frame: torch.Tensor,   # [B, T,   FRAME_DIM]
        delta: torch.Tensor,   # [B, T-1, DELTA_DIM]
    ) -> Dict[str, torch.Tensor]:
        out, i = {}, 0
        for k, dim in cls.FRAME_REPR.items():
            out[k] = frame[..., i: i + dim]
            i += dim
        i = 0
        for k, dim in cls.DELTA_REPR.items():
            out[k] = delta[..., i: i + dim]
            i += dim
        return out

    # ── Convenience: flat frame-only tensor (no deltas) ─────────────────────

    @classmethod
    def dict_to_tensor(cls, d: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Flat [B, T, FRAME_DIM] tensor — frame features only."""
        return torch.cat([d[k] for k in cls.FRAME_REPR], dim=-1)

    @classmethod
    def tensor_to_dict(cls, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Unpack a flat frame tensor back into a dict."""
        out, i = {}, 0
        for k, dim in cls.FRAME_REPR.items():
            out[k] = t[..., i: i + dim]
            i += dim
        return out

    # ── Rotation helpers ────────────────────────────────────────────────────

    @staticmethod
    def pack_poses_6d(
        global_orient: torch.Tensor,   # [B, T, 3, 3]
        body_pose:     torch.Tensor,   # [B, T, 21, 3, 3]
    ) -> torch.Tensor:
        """Pack both rotations into a single [B, T, 22*6] tensor."""
        B, T = global_orient.shape[:2]
        go_6d = rotmat_to_6d(global_orient)                         # [B, T, 6]
        bp_6d = rotmat_to_6d(body_pose).reshape(B, T, 21 * 6)      # [B, T, 126]
        return torch.cat([go_6d, bp_6d], dim=-1)                    # [B, T, 132]

    @staticmethod
    def unpack_poses_6d(
        poses_6d: torch.Tensor,        # [B, T, 132]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Unpack into global_orient [B,T,3,3] and body_pose [B,T,21,3,3]."""
        go_6d = poses_6d[..., :6]
        bp_6d = poses_6d[..., 6:]
        B, T  = poses_6d.shape[:2]
        global_orient = sixd_to_rotmat(go_6d)                           # [B, T, 3, 3]
        body_pose     = sixd_to_rotmat(bp_6d.reshape(B, T, 21, 6))      # [B, T, 21, 3, 3]
        return global_orient, body_pose

    # ── Canonicalization ────────────────────────────────────────────────────

    @staticmethod
    def canonicalize(
        seq: Dict[str, torch.Tensor],
        pelvis_world: Optional[torch.Tensor] = None,
        left_hip_world: Optional[torch.Tensor] = None,
        right_hip_world: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Transform the sequence into a body-centred canonical frame
        defined by the FIRST frame's pelvis and hips.

        If joint positions are not provided, the pelvis is approximated
        as transl[:, 0] and the hip direction is taken from global_orient.

        Args:
            seq: dict with transl [B,T,3], global_orient [B,T,3,3], body_pose [B,T,21,3,3]
            pelvis_world:    [B, 3]  first-frame pelvis position   (optional)
            left_hip_world:  [B, 3]  first-frame left-hip position  (optional)
            right_hip_world: [B, 3]  first-frame right-hip position (optional)

        Returns:
            transf_rotmat [B,3,3], transf_transl [B,1,3],
            seq (modified in-place with canonicalized values)
        """
        seq = deepcopy(seq)
        B, T, _ = seq["transl"].shape
        device   = seq["transl"].device

        if pelvis_world is None:
            # Fall back: use transl as pelvis proxy
            pelvis_world = seq["transl"][:, 0]             # [B, 3]

        if left_hip_world is None or right_hip_world is None:
            # Approximate hip direction from global_orient x-axis (first frame)
            # global_orient[:, 0] maps body-local → world for the pelvis
            go0 = seq["global_orient"][:, 0]               # [B, 3, 3]
            right_vec = go0[:, :, 0]                        # local x → world
            half_width = 0.1                                # rough hip half-width [m]
            right_hip_world = pelvis_world + half_width * right_vec
            left_hip_world  = pelvis_world - half_width * right_vec

        # Build canonical frame from first frame
        transf_rotmat, transf_transl = get_new_coordinate(
            pelvis_world, left_hip_world, right_hip_world
        )                                                   # [B,3,3], [B,1,3]
        R_inv = transf_rotmat.permute(0, 2, 1)             # [B,3,3]  world → local

        # ── Rotate global_orient ──────────────────────────────────────────
        # new_go = R_inv @ old_go
        seq["global_orient"] = torch.einsum(
            "bij,btjk->btik", R_inv, seq["global_orient"]
        )

        # ── Rotate + translate transl ─────────────────────────────────────
        # bring translation into local frame (subtract origin, rotate)
        seq["transl"] = torch.einsum(
            "bij,btj->bti", R_inv, seq["transl"] - transf_transl
        )

        # body_pose is relative to pelvis, no change needed (local rotations)

        return transf_rotmat, transf_transl, seq

    @staticmethod
    def transform_to_world(
        seq:           Dict[str, torch.Tensor],
        transf_rotmat: torch.Tensor,   # [B, 3, 3]
        transf_transl: torch.Tensor,   # [B, 1, 3]
    ) -> Dict[str, torch.Tensor]:
        """
        Inverse of canonicalize — bring a canonical sequence back to world space.
        """
        seq = deepcopy(seq)

        seq["transl"]= seq["transl"].to(transf_rotmat.device)
        # transl: rotate back then shift
        seq["transl"] = (
            torch.einsum("bij,btj->bti", transf_rotmat, seq["transl"])
            + transf_transl
        )
        seq["global_orient"] = seq["global_orient"].to(transf_rotmat.device)
        # global_orient: R_world @ R_canonical
        seq["global_orient"] = torch.einsum(
            "bij,btjk->btik", transf_rotmat, seq["global_orient"]
        )

        # body_pose unchanged (joint-local)
        return seq

    # ── Feature calculation ─────────────────────────────────────────────────

    @staticmethod
    def calc_features(seq: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute the full motion feature dict from a (canonicalized) sequence.

        Input seq keys: transl [B,T,3], global_orient [B,T,3,3], body_pose [B,T,21,3,3]

        Output features:
            transl                  [B, T, 3]
            poses_6d                [B, T, 132]    global_orient(6) + body_pose(21×6)
            transl_delta            [B, T-1, 3]
            global_orient_delta_6d  [B, T-1, 6]
        """
        B, T, _ = seq["transl"].shape

        poses_6d = BodyUtility.pack_poses_6d(
            seq["global_orient"], seq["body_pose"]
        )                                                           # [B, T, 132]

        transl_delta = seq["transl"][:, 1:] - seq["transl"][:, :-1]  # [B, T-1, 3]

        # relative global orient: R_t @ R_{t-1}^T
        go      = seq["global_orient"]                              # [B, T, 3, 3]
        go_rel  = torch.matmul(go[:, 1:], go[:, :-1].permute(0, 1, 3, 2))  # [B, T-1, 3, 3]
        go_delta_6d = rotmat_to_6d(go_rel)                         # [B, T-1, 6]

        return {
            "transl":                  seq["transl"],
            "poses_6d":                poses_6d,
            "transl_delta":            transl_delta,
            "global_orient_delta_6d":  go_delta_6d,
        }

    # ── Full pipeline helpers ────────────────────────────────────────────────

    @classmethod
    def encode_sequence(
        cls,
        seq: Dict[str, torch.Tensor],
        normalization_mode: str = "canonical",
        # ── "canonical" options ──────────────────────────────────────────────
        pelvis_world:    Optional[torch.Tensor] = None,  # [B, 3]
        left_hip_world:  Optional[torch.Tensor] = None,  # [B, 3]
        right_hip_world: Optional[torch.Tensor] = None,  # [B, 3]
        # ── "scene" options ──────────────────────────────────────────────────
        scene_origin:  Optional[torch.Tensor] = None,    # [B, 1, 3]  shared origin
        scene_rotmat:  Optional[torch.Tensor] = None,    # [B, 3, 3]  shared rotation
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Convert axis-angle → rotmat, normalise, then compute features.

        normalization_mode options
        ──────────────────────────
        "canonical"  (default)
            Full body-centred frame: remove first-frame position AND facing.
            Best for learning pose patterns in isolation, but can make short
            sequences appear near-static in canonical space.

        "position"
            Subtract first-frame pelvis position only; world orientation is
            kept intact.  Motion magnitudes are preserved and the model sees
            facing direction as part of the signal.  Good first alternative
            when "canonical" produces collapsed sequences.

        "scene"
            No per-primitive normalisation.  Use when both persons in an
            interaction share a common origin (e.g. mid-point of the pair)
            and you want to preserve relative spatial layout.
            Optionally pass scene_origin / scene_rotmat for a shared
            scene-level rigid transform that is applied identically to every
            primitive.  If neither is given, values are passed through raw.

        Returns
        ───────
        transf_rotmat  [B, 3, 3]   rotation applied   (identity if none)
        transf_transl  [B, 1, 3]   translation applied (zeros  if none)
        feature_dict
        """
        seq = deepcopy(seq)
        B      = seq["global_orient"].shape[0]
        S      = seq["global_orient"].shape[1]
        device = seq["transl"].device

        # ── axis-angle → rotation matrix ────────────────────────────────────
        seq["global_orient"] = transforms.axis_angle_to_matrix(seq["global_orient"])
        seq["body_pose"]     = transforms.axis_angle_to_matrix(
            seq["body_pose"].reshape(B, S, 21, 3)
        )

        # ── normalise ───────────────────────────────────────────────────────
        if normalization_mode == "canonical":
            # Remove first-frame position AND facing direction
            R, t, seq = cls.canonicalize(
                seq, pelvis_world, left_hip_world, right_hip_world
            )

        elif normalization_mode == "position":
            # Subtract first-frame position only; keep world orientation
            t = seq["transl"][:, :1, :]                              # [B, 1, 3]
            seq["transl"] = seq["transl"] - t
            R = torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1).clone()

        elif normalization_mode == "scene":
            if scene_origin is not None or scene_rotmat is not None:
                # Apply a shared rigid transform (same for all primitives in scene)
                t     = scene_origin if scene_origin is not None \
                        else torch.zeros(B, 1, 3, device=device)
                R     = scene_rotmat if scene_rotmat is not None \
                        else torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1).clone()
                R_inv = R.permute(0, 2, 1)
                seq["transl"]        = torch.einsum("bij,btj->bti", R_inv, seq["transl"] - t)
                seq["global_orient"] = torch.einsum("bij,btjk->btik", R_inv, seq["global_orient"])
            else:
                # Pure pass-through — no normalisation whatsoever
                t = torch.zeros(B, 1, 3, device=device)
                R = torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1).clone()

        else:
            raise ValueError(
                f"Unknown normalization_mode '{normalization_mode}'. "
                "Choose from: 'canonical', 'position', 'scene'."
            )

        features = cls.calc_features(seq)
        features["transf_rotmat"] = R
        features["transf_transl"] = t
        return R, t, features

    @classmethod
    def decode_features(
        cls,
        features:      Dict[str, torch.Tensor],
        transf_rotmat: torch.Tensor,   # [B, 3, 3]  identity if no rotation applied
        transf_transl: torch.Tensor,   # [B, 1, 3]  zeros  if no translation applied
    ) -> Dict[str, torch.Tensor]:
        """
        Inverse of encode_sequence — works identically for all three
        normalization modes because each mode stores its own R and t
        (identity / zeros when unused), so transform_to_world is always correct.
        """
        global_orient, body_pose = cls.unpack_poses_6d(features["poses_6d"])
        canon_seq = {
            "transl":        features["transl"],
            "global_orient": global_orient,
            "body_pose":     body_pose,
        }
        world_seq = cls.transform_to_world(canon_seq, transf_rotmat, transf_transl)

        go_aa = transforms.matrix_to_axis_angle(world_seq["global_orient"])   # [B,T,3]
        bp_aa = transforms.matrix_to_axis_angle(
            world_seq["body_pose"].reshape(-1, 3, 3)
        ).reshape(world_seq["body_pose"].shape[0],
                  world_seq["body_pose"].shape[1], 21 * 3)                    # [B,T,63]

        return {
            "transl":        world_seq["transl"],
            "global_orient": go_aa,
            "body_pose":     bp_aa,
        }

    # ── Batch slicing ────────────────────────────────────────────────────────

    @staticmethod
    def get_batch_item(seq: Dict[str, torch.Tensor], idx: int) -> Dict[str, torch.Tensor]:
        return {k: (v[idx] if isinstance(v, torch.Tensor) else v) for k, v in seq.items()}


# ─────────────────────────────────────────────────────────────
#  Sanity check
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    B, T = 4, 20
    device = "cpu"

    def make_seq():
        return {
            "transl":        torch.randn(B, T, 3) * 2,           # non-trivial translation
            "global_orient": torch.randn(B, T, 3),               # axis-angle
            "body_pose":     torch.randn(B, T, 63),              # axis-angle flat
        }

    # ── Mode 1: canonical (default) ──────────────────────────
    print("═" * 50)
    print("Mode: canonical")
    seq = make_seq()
    R, t, features = BodyUtility.encode_sequence(seq, normalization_mode="canonical")
    print(f"  transl after canon — mean: {features['transl'].mean():.4f}  "
          f"std: {features['transl'].std():.4f}")
    world = BodyUtility.decode_features(features, R, t)
    err = (world["transl"] - transforms.axis_angle_to_matrix(
        torch.zeros_like(seq["transl"])   # placeholder — check via re-encode
    )).abs().max()
    # round-trip check
    seq2 = make_seq()
    R2, t2, f2 = BodyUtility.encode_sequence(seq2, normalization_mode="canonical")
    w2 = BodyUtility.decode_features(f2, R2, t2)
    seq2_rotmat = transforms.axis_angle_to_matrix(seq2["global_orient"])
    print(f"  round-trip transl err:  {(w2['transl'] - seq2['transl']).abs().max():.2e}")
    print(f"  frame_tensor: {BodyUtility.dict_to_tensor(f2).shape}")

    # ── Mode 2: position only ─────────────────────────────────
    print("\n" + "═" * 50)
    print("Mode: position")
    seq = make_seq()
    R, t, features = BodyUtility.encode_sequence(seq, normalization_mode="position")
    print(f"  transl[0, 0] after position norm: {features['transl'][0, 0]}  (should be ~0)")
    print(f"  transl std (should be > canonical): {features['transl'].std():.4f}")
    w = BodyUtility.decode_features(features, R, t)
    print(f"  round-trip transl err:  {(w['transl'] - seq['transl']).abs().max():.2e}")
    print(f"  frame_tensor: {BodyUtility.dict_to_tensor(features).shape}")

    # ── Mode 3: scene (no normalisation) ─────────────────────
    print("\n" + "═" * 50)
    print("Mode: scene (pass-through)")
    seq = make_seq()
    R, t, features = BodyUtility.encode_sequence(seq, normalization_mode="scene")
    print(f"  transl[0, 0] (world position preserved): {features['transl'][0, 0].tolist()}")
    w = BodyUtility.decode_features(features, R, t)
    print(f"  round-trip transl err:  {(w['transl'] - seq['transl']).abs().max():.2e}")

    # ── Mode 3b: scene with shared transform ─────────────────
    print("\n" + "═" * 50)
    print("Mode: scene (shared origin for two-person interaction)")
    seq_a, seq_b = make_seq(), make_seq()
    # Compute scene origin as midpoint of both persons' first frames
    mid = (seq_a["transl"][:, :1] + seq_b["transl"][:, :1]) / 2.0   # [B, 1, 3]
    shared_origin = mid
    R_a, t_a, fa = BodyUtility.encode_sequence(
        seq_a, normalization_mode="scene", scene_origin=shared_origin
    )
    R_b, t_b, fb = BodyUtility.encode_sequence(
        seq_b, normalization_mode="scene", scene_origin=shared_origin
    )
    print(f"  person A transl[0,0]: {fa['transl'][0,0].tolist()}")
    print(f"  person B transl[0,0]: {fb['transl'][0,0].tolist()}")
    print(f"  (both relative to shared midpoint)")
    wa = BodyUtility.decode_features(fa, R_a, t_a)
    wb = BodyUtility.decode_features(fb, R_b, t_b)
    print(f"  round-trip err A: {(wa['transl'] - seq_a['transl']).abs().max():.2e}")
    print(f"  round-trip err B: {(wb['transl'] - seq_b['transl']).abs().max():.2e}")