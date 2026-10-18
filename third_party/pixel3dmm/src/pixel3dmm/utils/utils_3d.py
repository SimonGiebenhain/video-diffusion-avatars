from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

def backproject(depth_maps, normal_maps, Ks, Es, rgb=None, masks=None):
    points3d = {}
    normals3d = {}
    rgb3d = {}
    for cam_id in depth_maps.keys():
        depth_map = depth_maps[cam_id]
        normal_map = normal_maps[cam_id]


        ys = np.arange(depth_map.shape[0])
        xs = np.arange(depth_map.shape[1])
        p_screen = np.dstack(np.meshgrid(xs, ys, [1])).reshape((-1, 3))
        depth_mask = (depth_map > 0)  & (depth_map < 1.4)
        if masks is not None:
            # upsample mask
            I = Image.fromarray(masks[cam_id])
            I = I.resize((I.size[0]*2, I.size[1]*2))
            depth_mask = np.logical_and(depth_mask, np.array(I).astype(np.bool))
        depths = depth_map[depth_mask]
        p_screen = p_screen[depth_mask.reshape(-1)]
        p_screen_canonical = p_screen @ Ks[cam_id].invert().T
        p_cam = p_screen_canonical * np.expand_dims(depths, 1)
        p_cam_hom = np.hstack([p_cam, np.ones((p_cam.shape[0], 1))])
        p_world = p_cam_hom @ Es[cam_id].T
        ns = np.ones_like(p_world)
        ns[:, :3] = normal_map[depth_mask]
        n_world = ns @ Es[cam_id].T

        points3d[cam_id] = p_world[:, :3]
        normals3d[cam_id] = n_world[:, :3]
        if rgb is not None:
            rgb_lin = rgb[cam_id].reshape((-1, 3))
            rgb_valid = rgb_lin[depth_mask.reshape(-1)]
            rgb3d[cam_id] = rgb_valid

    if rgb is None:
        return points3d, normals3d
    else:
        return points3d, normals3d, rgb3d


def get_view_dirs(Ks, Es, image_shape, rgb=None, masks=None):
    points3d = {}
    view_dirs = {}
    for cam_id in Ks.keys():
        ys = np.arange(image_shape[0])
        xs = np.arange(image_shape[1])
        p_screen = np.dstack(np.meshgrid(xs, ys, [1])).reshape((-1, 3))
        if masks is not None:
            # upsample mask
            I = Image.fromarray(masks[cam_id])
            I = I.resize((I.size[0]*2, I.size[1]*2))
            depth_mask = np.logical_and(depth_mask, np.array(I).astype(np.bool))
        p_screen = np.reshape(p_screen, [-1, 3])
        p_screen_canonical = p_screen @ Ks[cam_id].invert().T
        p_cam = p_screen_canonical * 1
        p_cam_hom = np.hstack([p_cam, np.ones((p_cam.shape[0], 1))])
        p_world = p_cam_hom @ Es[cam_id].T

        points3d[cam_id] = p_world[:, :3]

        origin = Es[cam_id][:3, 3]
        view_dirs[cam_id] = p_world[:, :3] - origin
        view_dirs[cam_id] /= np.linalg.norm(view_dirs[cam_id], axis=-1, keepdims=True)

        #if rgb is not None:
        #    rgb_lin = rgb[cam_id].reshape((-1, 3))
        #    rgb_valid = rgb_lin[depth_mask.reshape(-1)]
        #    rgb3d[cam_id] = rgb_valid

    #if rgb is None:
    #    return points3d, normals3d
    #else:

    return view_dirs





    return


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    # return functools.reduce(torch.matmul, matrices)
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])