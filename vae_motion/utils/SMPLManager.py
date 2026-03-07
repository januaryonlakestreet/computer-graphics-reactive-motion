import torch
import smplx
from torch.cuda.amp import autocast


class SMPLManager(torch.nn.Module):
    def __init__(self, model_path="models/", gender="neutral", device="cuda"):
        super().__init__()
        self.device = device
        self.model = smplx.create(
            model_path=model_path,
            model_type="smplx",
            gender=gender,
            use_pca=False,
            num_betas=16,
            num_expression_coeffs=10,
        ).to(device)

        # buffers
        self.register_buffer("zero_jaw", torch.zeros(1, 3, device=device))
        self.register_buffer("zero_handL", torch.zeros(1, 45, device=device))
        self.register_buffer("zero_handR", torch.zeros(1, 45, device=device))
        self.register_buffer("zero_expr", torch.zeros(1, 10, device=device))
        self.register_buffer("zero_eyeL", torch.zeros(1, 3, device=device))
        self.register_buffer("zero_eyeR", torch.zeros(1, 3, device=device))
        self.register_buffer("Betas", torch.zeros(1, 16, device=device))

    def smpl_forward(self, body_pose, global_orient, transl, return_verts=True, chunk_size=128):

        """
        Run SMPL forward in mixed precision and chunks over time to save memory.
        Args:
            betas:        (B, 16)
            body_pose:    (B, T, 63)
            global_orient:(B, T, 3)
            transl:       (B, T, 3)
            B, T:         batch and time dims
            return_verts: if False, only joints are returned
            chunk_size:   number of (batch*time) frames per forward
        Returns:
            joints: (B, T, J, 3)
            verts:  (B, T, V, 3) or None
        """

        device = body_pose.device
        B = body_pose.shape[0]
        T = body_pose.shape[1]
        BT = B * T
        body_pose_bt = body_pose.reshape(BT, -1)
        global_orient_bt = global_orient.reshape(BT, -1)
        transl_bt = transl.reshape(BT, -1)
        #betas_bt = betas.unsqueeze(1).expand(-1, T, -1).reshape(BT, -1)

        # expand static buffers
        jaw = self.zero_jaw.expand(BT, -1)
        handL = self.zero_handL.expand(BT, -1)
        handR = self.zero_handR.expand(BT, -1)
        expr = self.zero_expr.expand(BT, -1)
        eyeL = self.zero_eyeL.expand(BT, -1)
        eyeR = self.zero_eyeR.expand(BT, -1)

        betas_bt = self.Betas.expand(BT,-1)
        joints_list, verts_list = [], []

        for start in range(0, BT, chunk_size):
            end = min(start + chunk_size, BT)

            with torch.amp.autocast('cuda', enabled=True):
                # Ensure the model output remains on the GPU (default behavior)
                out = self.model(
                    betas=betas_bt[start:end],
                    body_pose=body_pose_bt[start:end],
                    global_orient=global_orient_bt[start:end],
                    transl=transl_bt[start:end],
                    jaw_pose=jaw[start:end],
                    left_hand_pose=handL[start:end],
                    right_hand_pose=handR[start:end],
                    expression=expr[start:end],
                    leye_pose=eyeL[start:end],
                    reye_pose=eyeR[start:end],
                    return_verts=return_verts,
                    return_joints=True,
                )

            # Remove .cpu() and ensure .float() (or remove it if autocast handles it)
            # Keeping .float() for explicit consistency
            joints_list.append(out.joints.float())
            if return_verts:
                verts_list.append(out.vertices.float())

        # Concatenation occurs on the GPU, resulting in GPU tensors
        joints = torch.cat(joints_list, dim=0).view(B, T, -1, 3)
        verts = None
        if return_verts:
            verts = torch.cat(verts_list, dim=0).view(B, T, -1, 3)

        return joints, verts, out
