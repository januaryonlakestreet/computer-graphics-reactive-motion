import torch


def batched_quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Converts batches of rotations given as quaternions to the vectorized axis-angle
    representation. This function is numerically stable near identity rotations.

    Args:
        quaternions: A tensor of quaternions (w, x, y, z) with the real part first,
                      with shape [..., 4]. The final axis/angle vector will
                      have shape [..., 3].

    Returns:
        Rotations given as a vector in axis-angle form, with shape [..., 3],
        where the magnitude is the angle turned anticlockwise in radians
        around the vector's direction.
    """
    # 1. Separate the vector part (x, y, z) and the scalar part (w)
    # The vector part (u) is at indices 1, 2, 3. Shape [batch_size, 22, 3]
    vector_part = quaternions[..., 1:]

    # The scalar part (w) is at index 0. Shape [batch_size, 22, 1]
    scalar_part = quaternions[..., :1]

    # 2. Calculate the norm of the vector part: |u| = sin(theta/2)
    # norms shape: [batch_size, 22, 1]
    norms = torch.norm(vector_part, p=2, dim=-1, keepdim=True)

    # 3. Calculate the half-angle: theta/2 = atan2(|u|, w)
    # half_angles shape: [batch_size, 22, 1]
    half_angles = torch.atan2(norms, scalar_part)

    # 4. Calculate the stabilizing factor: sin(theta/2) / theta
    # sin_half_angles_over_angles = (sin(theta/2) / (theta/2)) * 0.5 * (2/pi) * pi
    # The torch.sinc(x) calculates sin(pi*x) / (pi*x). We use this to implement:
    # sin(theta/2) / (theta/2) = sinc( (theta/2) / pi )

    # The factor required is sin(theta/2) / theta
    # factor_reciprocal = sin(theta/2) / theta
    factor_reciprocal = 0.5 * torch.sinc(half_angles / torch.pi)

    # --- Handling the Degenerate Case (Near Identity) ---
    # When theta is near 0, factor_reciprocal is near 0.5.
    # We enforce a minimum value for stability against potential division by zero.
    # The reciprocal of the factor (theta / sin(theta/2)) is what we'll multiply by.

    # Use a small tolerance for the factor reciprocal itself
    epsilon = torch.finfo(quaternions.dtype).eps

    # Clamp to ensure the denominator is never zero.
    # For angle=0, factor_reciprocal is 0.5.
    factor_reciprocal = factor_reciprocal.clamp_min(epsilon)

    # 5. Calculate the final axis-angle vector: (theta * v) = u / (sin(theta/2) / theta)
    # axis_angle_vector shape: [batch_size, 22, 3]
    axis_angle_vector = vector_part / factor_reciprocal

    return axis_angle_vector
def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    sin_half_angles_over_angles = 0.5 * torch.sinc(half_angles / torch.pi)

    return quaternions[..., 1:] / sin_half_angles_over_angles