import torch
from nvf.env.utils import pose2tensor, tensor2pose
import torch.nn.functional as F
from nvf.visibility.visibility import get_density, single_camera_field_opacity
import numpy as np
class BaseSampler():
    def __init__(self, cfg=None):
        self.cfg = cfg
        self.pipeline = None
        # self.camera_aabb = torch.tensor([[-3, 3], [-3, 3], [-3, 3]])
        if cfg.camera_aabb.shape == (2,3):
            self.camera_aabb = cfg.camera_aabb.T
        elif cfg.camera_aabb.shape == (3,2):
            self.camera_aabb = cfg.camera_aabb
        else:
            raise ValueError('camera_aabb should be of shape (2,3) or (3,2)')
        
        if cfg.target_aabb.shape == (2,3):
            self.target_aabb = cfg.target_aabb.T
        elif cfg.target_aabb.shape == (3,2):
            self.target_aabb = cfg.target_aabb
        else:
            raise ValueError('target_aabb should be of shape (2,3) or (3,2)')
        
        self.pipeline = None
        self.density_threshold = cfg.density_threshold
        # breakpoint()
        
    def setup(self, scale=1):
        pass
    
    def __call__(self,*args, **kwargs):
        return self.sample(*args, **kwargs)
    
    def filter_poses(self, poses):
        poses = self.filter_valid_quat(poses)
        if self.pipeline is None or self.density_threshold is None:
            return poses
        else:
            positions = poses[:, -3:]
            nerf_model = self.pipeline.trainer.pipeline.model
            density = get_density(positions.to(device=nerf_model.device), nerf_model).squeeze().to(poses)
            # breakpoint()
            valid_idx = density<self.density_threshold
            # print('valid sample:', valid_idx.sum())
            poses = poses[valid_idx, :]
            return poses
    
    def filter_valid_quat(self, result):
        quat = result[:, :4]
        # filter out invalid poses

        quat_norms = torch.norm(quat, dim=1)
        valid_quat_indices = quat_norms > 1e-9
        result = result[valid_quat_indices]

        valid_indices = torch.all(~torch.isnan(result), dim=1)
        result = result[valid_indices]
        return result
    
    def sample(self, n_sample, *args, **kwargs):
        # print('Base sampling')
        if self.pipeline is not None and self.cfg.check_density:
            
            n_sample_oragin = n_sample
            n_sample = int(n_sample*1.2)

            result = sample_poses(n_sample, bounds = self.camera_aabb)
            
            result = self.filter_poses(result)

            result = result[:min(n_sample_oragin,result.shape[0]),:]
            return result
        else:
            return sample_poses(n_sample, bounds = self.camera_aabb)

class CFSampler(BaseSampler):

    def __init__(self, cfg=None):
        super().__init__(cfg)

        self.visibility_threshold = 0.95

    def filter_poses_collision(self, poses, current_pose):
        positions = poses[:, -3:]
        c2w = torch.tensor(current_pose.matrix()).cuda()
        nerf_model = self.pipeline.trainer.pipeline.model

        opacity = single_camera_field_opacity(
            points=positions.to(device=nerf_model.device), 
            pose=c2w, 
            model=nerf_model
        ).squeeze().to(poses)

        visbility = 1 - opacity

        valid_idx = visbility > self.visibility_threshold
        poses = poses[valid_idx, :]

        return poses

    def __call__(self, *args, **kwargs):
        return self.sample(*args, **kwargs)

    def sample(self, n_sample, pose, *args, **kwargs):

        n_sample_origin = n_sample
        n_sample = int(n_sample*3)

        result = sample_poses(n_sample, bounds = self.camera_aabb)
        
        result = self.filter_poses(result)
        result = self.filter_poses_collision(poses=result, current_pose=pose)

        result = result[:min(n_sample_origin,result.shape[0]),:]
        return result



def sample_poses(n_sample, bounds = [[-4, 4], [-4, 4], [-4,4.]]):
    """
    Sample random SE(3) poses.

    Parameters:
    - n_sample: number of samples
    - bounds: list of bounds for x, y, z e.g., [(x_min, x_max), (y_min, y_max), (z_min, z_max)]

    Returns:
    - A tensor of size n_sample x 7 with each row representing a pose as [qx, qy, qz, qw, x, y, z]
    """

    # Sample translations within the bounds
    translations = torch.stack([
        torch.FloatTensor(n_sample).uniform_(bounds[0][0], bounds[0][1]),
        torch.FloatTensor(n_sample).uniform_(bounds[1][0], bounds[1][1]),
        torch.FloatTensor(n_sample).uniform_(bounds[2][0], bounds[2][1]),
    ], dim=-1)

    # Sample unit quaternions for orientations
    quats = torch.randn(n_sample, 4)
    quats /= quats.norm(p=2, dim=1, keepdim=True)

    # Concatenate quaternions and translations to get the final pose
    poses = torch.cat([quats, translations], dim=1)

    return poses

def rotation_matrix_to_quaternion(rot_mats):
    """

    Returns:
    - torch.Tensor: A batch of quaternions of shape (batch_size, 4). in order x, y, z, w
    """
    batch_size = rot_mats.shape[0]
    
    q = torch.zeros((batch_size, 4), device=rot_mats.device)
    
    q[:, 3] = 0.5 * torch.sqrt(1 + rot_mats[:, 0, 0] + rot_mats[:, 1, 1] + rot_mats[:, 2, 2])
    q[:, 0] = torch.sign(rot_mats[:, 2, 1] - rot_mats[:, 1, 2]) * 0.5 * torch.sqrt(1 + rot_mats[:, 0, 0] - rot_mats[:, 1, 1] - rot_mats[:, 2, 2])
    q[:, 1] = torch.sign(rot_mats[:, 0, 2] - rot_mats[:, 2, 0]) * 0.5 * torch.sqrt(1 - rot_mats[:, 0, 0] + rot_mats[:, 1, 1] - rot_mats[:, 2, 2])
    q[:, 2] = torch.sign(rot_mats[:, 1, 0] - rot_mats[:, 0, 1]) * 0.5 * torch.sqrt(1 - rot_mats[:, 0, 0] - rot_mats[:, 1, 1] + rot_mats[:, 2, 2])
    return q

def pose_point_to_batch(t, target_points):
    ez = -F.normalize(target_points - t, dim=1)

    up = F.normalize(torch.rand(target_points.shape[0], 3), dim=1)
  
    ex = torch.cross(up, ez)
    ex = F.normalize(ex, dim=1)
    ey = torch.cross(ez, ex)
    ey = F.normalize(ey)
    # print(np.linalg.norm(ex), np.linalg.norm(ey), np.linalg.norm(ez))
    rot = torch.stack([ex, ey, ez], dim=2)

    quat = rotation_matrix_to_quaternion(rot)

    result = torch.cat([quat, t], dim=1)
    return result

def perturb_se3_pose_within_bounds(n_sample, pose_tensor, bounds = [[-5, 5], [-5, 5], [-5,5.]]):
    # Extract quaternion and translation from input tensor
    quat_original = pose_tensor[:4]
    trans_original = pose_tensor[4:]
    
    # Define max perturbation values
    max_roll, max_pitch, max_yaw = 0.1, 0.1, 0.1
    max_translation = 0.5
    
    # Generate perturbations for rotation (Euler angles) and translation
    perturbed_roll = (torch.rand(n_sample) * 2 - 1) * max_roll
    perturbed_pitch = (torch.rand(n_sample) * 2 - 1) * max_pitch
    perturbed_yaw = (torch.rand(n_sample) * 2 - 1) * max_yaw
    perturbed_translation = (torch.rand(n_sample, 3) * 2 - 1) * max_translation
    
    # Convert perturbed Euler angles to quaternion
    cos_roll = torch.cos(perturbed_roll / 2)
    sin_roll = torch.sin(perturbed_roll / 2)
    cos_pitch = torch.cos(perturbed_pitch / 2)
    sin_pitch = torch.sin(perturbed_pitch / 2)
    cos_yaw = torch.cos(perturbed_yaw / 2)
    sin_yaw = torch.sin(perturbed_yaw / 2)
    
    perturbed_quat = torch.zeros(n_sample, 4)
    perturbed_quat[:, 0] = sin_roll * cos_pitch * cos_yaw - cos_roll * sin_pitch * sin_yaw
    perturbed_quat[:, 1] = cos_roll * sin_pitch * cos_yaw + sin_roll * cos_pitch * sin_yaw
    perturbed_quat[:, 2] = cos_roll * cos_pitch * sin_yaw - sin_roll * sin_pitch * cos_yaw
    perturbed_quat[:, 3] = cos_roll * cos_pitch * cos_yaw + sin_roll * sin_pitch * sin_yaw
    
    # Multiply original quaternion with perturbed quaternion
    q1 = quat_original
    q2 = perturbed_quat
    w1, x1, y1, z1 = q1[3], q1[0], q1[1], q1[2]
    w2, x2, y2, z2 = q2[:, 3], q2[:, 0], q2[:, 1], q2[:, 2]
    combined_quat = torch.zeros(n_sample, 4)
    combined_quat[:, 0] = x1 * w2 + y1 * z2 - z1 * y2 + w1 * x2
    combined_quat[:, 1] = -x1 * z2 + y1 * w2 + z1 * x2 + w1 * y2
    combined_quat[:, 2] = x1 * y2 - y1 * x2 + z1 * w2 + w1 * z2
    combined_quat[:, 3] = -x1 * x2 - y1 * y2 - z1 * z2 + w1 * w2
    
    # Add perturbed translation to original translation
    combined_translation = perturbed_translation + trans_original
    
    # Clip the translations to stay within bounds
    for i in range(3):
        combined_translation[:, i] = torch.clamp(combined_translation[:, i], bounds[i][0], bounds[i][1])
    
    # Combine quaternion and translation to form SE(3) pose
    combined_pose = torch.cat([combined_quat, combined_translation], dim=1)
    
    return combined_pose