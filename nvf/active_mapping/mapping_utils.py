from numpy.typing import NDArray
import torch
import numpy as np
from typing import Any, Callable, Dict, List, Literal, Optional



def quat_to_rot_transform_torch(pose):
    """
    Convert a quaternion to a rotation matrix in PyTorch.

    Args:
        pose (Tensor): A tensor representing the quaternion ( x, y, z, w).

    Returns:
        Tensor: A 3x3 rotation matrix.
    """
    # rotation_matrix = quaternion_to_rotation_matrix(quaternion)
    Q = pose[0:4]
    T = pose[4:]
    # Ensure the Q is normalized
    # Q /= torch.norm(Q, p=2)
    transpose = torch.eye(4)

    # First row of the rotation matrix
    transpose[0][0] = 2 * (Q[3] * Q[3] + Q[0] * Q[0]) - 1
    transpose[0][1] = 2 * (Q[0] * Q[1] - Q[3] * Q[2])
    transpose[0][2] = 2 * (Q[0] * Q[2] + Q[3] * Q[1])
    
    # Second row of the rotation matrix
    transpose[1][0] = 2 * (Q[0] * Q[1] + Q[3] * Q[2])
    transpose[1][1] = 2 * (Q[3] * Q[3] + Q[1] * Q[1]) - 1
    transpose[1][2] = 2 * (Q[1] * Q[2] - Q[3] * Q[0])
    
    # Third row of the rotation matrix
    transpose[2][0] = 2 * (Q[0] * Q[2] - Q[3] * Q[1])
    transpose[2][1] = 2 * (Q[1] * Q[2] + Q[3] * Q[0])
    transpose[2][2] = 2 * (Q[3] * Q[3] + Q[2] * Q[2]) - 1

    transpose[0,3] = T[0]
    transpose[1,3] = T[1]
    transpose[2,3] = T[2]
        
    return transpose

def rotmat_to_quat_transform_torch(rot_matrix):
    # Ensure the input rotation matrix is a 3x3 matrix
    if rot_matrix.shape != (3, 3):
        raise ValueError("Input rotation matrix must be a 3x3 matrix")

    # Extract individual elements from the rotation matrix
    r11, r12, r13 = rot_matrix[0, 0], rot_matrix[0, 1], rot_matrix[0, 2]
    r21, r22, r23 = rot_matrix[1, 0], rot_matrix[1, 1], rot_matrix[1, 2]
    r31, r32, r33 = rot_matrix[2, 0], rot_matrix[2, 1], rot_matrix[2, 2]

    # Calculate quaternion components
    qw = 0.5 * torch.sqrt(1 + r11 + r22 + r33)
    qx = (r32 - r23) / (4 * qw)
    qy = (r13 - r31) / (4 * qw)
    qz = (r21 - r12) / (4 * qw)

    quaternion = torch.tensor([qw, qx, qy, qz])
    return quaternion

def to_transform(pose):
    """Convert quaternion & translation vector into transformation matrix"""
    transform = quat_to_rot_transform_torch(pose)
    return transform

def to_transform_np(pose: NDArray) -> np.ndarray:
    """Convert quaternion & translation (numpy) vector into transformation matrix"""

    Q = pose[0:4]
    T = pose[4:]
    rot_mat = Rotation.from_quat(Q).as_matrix()
    rotate = np.identity(4)
    rotate[0:3, 0:3] = rot_mat

    transpose = np.identity(4)
    transpose[0,3] = T[0]
    transpose[1,3] = T[1]
    transpose[2,3] = T[2]

    transform = np.matmul(transpose, rotate)

    return transform

def get_entropy_for_camera_ray_bundle_diff(model, camera_ray_bundle):
        num_rays_per_chunk = model.config.eval_num_rays_per_chunk
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        entropy_list = []
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.flatten()[start_idx:end_idx]
            # Dont calculate gradient over samples
            # Use indices to sample a bunch of rays from bundle
            indices = torch.randperm(ray_bundle.shape[0])[:int(ray_bundle.shape[0]/10)]
            indices.requires_grad=False
                
            outputs = model.get_entropy(ray_bundle=ray_bundle[indices])
            # for output_name, output in outputs.items():  # type: ignore
            #     if not torch.is_tensor(output):
            #         # TODO: handle lists of tensors as well
            #         continue
            entropy_list.append(outputs)
            entropy = torch.cat(entropy_list)
        return entropy

@torch.no_grad()
def get_entropy_for_camera_ray_bundle(model, camera_ray_bundle):
        num_rays_per_chunk = model.config.eval_num_rays_per_chunk
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        entropy_list = []
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            outputs = model.get_entropy(ray_bundle=ray_bundle)
            # for output_name, output in outputs.items():  # type: ignore
            #     if not torch.is_tensor(output):
            #         # TODO: handle lists of tensors as well
            #         continue
            entropy_list.append(outputs)

        entropy = torch.cat(entropy_list).view(image_height, image_width, -1)  # type: ignore
        return entropy

def get_camera_dict(
    transform,
    fov,
    height: int = 300,
    width: int = 300
) -> Dict:
    return  {
                "camera_type": "perspective",
                "render_height": height,
                "render_width": width,
                "camera_path": [
                {
                    "camera_to_world": transform.flatten(),
                    "fov": fov,
                    "aspect": 1,#1.455543358946213
                }
                ],
            }
