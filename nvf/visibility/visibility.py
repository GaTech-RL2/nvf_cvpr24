import glob
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

import nerfacc

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras.rays import Frustums, RayBundle, RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.ray_samplers import (UniformSampler,
                                                      VolumetricSampler)
# from nerfstudio.scripts.pipeline import ActiveMapper, to_transform
from nerfstudio.utils.eval_utils import eval_load_checkpoint, eval_setup

use_uniform = True
# print(f'gt visibility uniform sampler: {use_uniform}')

def generate_ray_from_points(points, pose):
    '''
    points: n x 3
    pose: 4 x 4
    '''
    origin = pose[:3,3]
    origin = origin.repeat(points.shape[0], 1)

    directions = points - origin
    distance = torch.norm(directions, dim=-1, keepdim=True)
    directions = directions / distance

    pixel_area = torch.ones(points.shape[0], 1).to(points)

    camera_indices = torch.zeros(points.shape[0], 1, dtype=int)

    nears = torch.zeros(points.shape[0], 1).to(points)
    fars = distance.view(-1,1).to(points)

    return RayBundle(origins=origin, directions=directions, pixel_area=pixel_area, camera_indices=camera_indices, nears=nears, fars=fars)

@torch.no_grad()
def single_camera_field_opacity(points, pose, model):
    
    ray_bundle = generate_ray_from_points(points, pose)
    
    sampler = UniformSampler(
                num_samples = 200,
            )

    ray_samples = sampler(ray_bundle)

    # uniform_field_outputs = model.field(ray_samples)
    # opacity = ray_samples.get_opacity(uniform_field_outputs[FieldHeadNames.DENSITY])
    density, _ = model.field.get_density(ray_samples)
    opacity = ray_samples.get_opacity(density)
    opacity = opacity[:,-1,0]
    # breakpoint()
    return opacity

@torch.no_grad()
def single_camera_field_opacity_adaptive_sampler(points, pose, model):
    
    ray_bundle = generate_ray_from_points(points, pose)
    ray_bundle.camera_indices = ray_bundle.camera_indices.to(model.device)
    distance = ray_bundle.fars
    ray_bundle.fars = distance * (1 - 1/200.)
    num_rays = len(ray_bundle)
    
    # breakpoint()

    ray_samples, ray_indices = model.sampler(
                ray_bundle=ray_bundle,
                near_plane=model.config.near_plane,
                far_plane=model.config.far_plane,
                render_step_size=model.config.render_step_size,
                alpha_thre=model.config.alpha_thre,
                cone_angle=model.config.cone_angle,
            )
    
    pixel_area = torch.zeros(points.shape[0], 1).to(points)
    ray_indices = torch.cat([ray_indices, torch.arange(num_rays).to(ray_indices)], dim=0)
    rank = torch.argsort(ray_indices)
    ray_indices = ray_indices[rank]
    # breakpoint()
    origins = torch.cat([ray_samples.frustums.origins, ray_bundle.origins], dim=0)[rank,...]
    directions = torch.cat([ray_samples.frustums.directions, ray_bundle.directions], dim=0)[rank,...]
    starts = torch.cat([ray_samples.frustums.starts, ray_bundle.fars], dim=0)[rank,...]
    ends = torch.cat([ray_samples.frustums.ends, 2*distance - ray_bundle.fars], dim=0)[rank,...]
    pixel_area = torch.cat([ray_samples.frustums.pixel_area, pixel_area], dim=0)[rank,...]

    # breakpoint()
    frustums = Frustums(origins=origins, directions=directions, starts=starts, ends=ends, pixel_area=pixel_area)

    camera_indices = torch.cat([ray_samples.camera_indices, ray_bundle.camera_indices], dim=0)[rank,...]
    # frustums = Frustums(origins=origins, directions=ray_bundle.directions, starts=ray_bundle.fars-0.01, ends=ray_bundle.fars+0.01, pixel_area=pixel_area)
    # camera_indices = torch.zeros(points.shape[0], 1, dtype=int, device=points.device)
    ray_samples = RaySamples(frustums, camera_indices=camera_indices)

    
    # new_ray_indices = torch.arange(num_rays).to(ray_indices)

    # ray_samples = torch.concat([ray_samples, new_ray_samples])

    # ray_samples = sampler(ray_bundle)
    density, _ = model.field.get_density(ray_samples)

    packed_info = nerfacc.pack_info(ray_indices, num_rays)
    #render_transmittance_from_density
    #render_visibility_from_density
    trans = nerfacc.render_transmittance_from_density(
        t_starts=ray_samples.frustums.starts[..., 0],
        t_ends=ray_samples.frustums.ends[..., 0],
        sigmas=density[..., 0],
        packed_info=packed_info,
        # early_stop_eps = 0.5
    )[0]

    # 
    # vis_ray = torch.zeros(num_rays).to(density)
    # opacity = vis_ray.scatter_reduce(0, ray_indices, trans, 'amax')
    # breakpoint()
    # trans = trans.to(density)
    # trans = torch.where(trans<=0.99999, 0., 1.)
    
    # opacity = 1- trans[-num_rays:]
    # breakpoint()
    vis_ray = torch.ones(num_rays).to(density)
    opacity = 1- vis_ray.scatter_reduce(0, ray_indices, trans, 'amin')
    # breakpoint()
    # mask = torch.zeros(num_rays).to(density)
    # mask = mask.scatter_reduce(0, ray_indices, torch.ones_like(trans), 'sum')
    # opacity[ mask<1] = 0.

    # breakpoint()
    # vis_ray = torch.ones(num_rays).to(density)
    # for i in ray_indices:
    #     vis_ray[i] =torch.min(trans[i], vis_ray[i])
    # opacity = 1-vis_ray


    # uniform_field_outputs = model.field(ray_samples)
    # opacity = ray_samples.get_opacity(uniform_field_outputs[FieldHeadNames.DENSITY])
    # density, _ = model.field.get_density(ray_samples)
    # opacity = ray_samples.get_opacity(density)
    # opacity = opacity[:,-1,0]
    # breakpoint()
    return opacity

def fov_visibility(cameras, points, z_min=0, z_max=10):
    # Convert points to homogeneous coordinates
    K = cameras.get_intrinsics_matrices().to(points)

    width = cameras.width
    height = cameras.height

    c2w = cameras.camera_to_worlds
    num_cameras = c2w.size(0)
    num_points = points.size(0)
    
    # c2w = torch.cat([c2w, torch.zeros(num_cameras, 1, 4, device=c2w.device)], dim=1)  # num_cameras x 4 x 4
    # c2w[:, 3, 3] = 1.0
    # points_expanded = points.unsqueeze(0).expand(num_cameras, -1, -1)
    # Homogeneous coordinates for points
    # points_h = torch.cat([points_expanded, torch.ones(num_cameras, num_points, 1, dtype=points.dtype)], dim=2)  # num_points x 4
    points_h = torch.cat([points, torch.ones(num_points, 1, device=points.device)], dim=1) 
    # Extract rotation and translation from c2w
    R = c2w[:, :, :3]
    t = c2w[:, :, 3:]
    
    # Compute world-to-camera transformation matrices
    R_transpose = R.permute(0, 2, 1)
    t_new = torch.bmm(-R_transpose, t)
    w2c = torch.cat([R_transpose, t_new], dim=2)
    
    # Transform points to camera frame: num_cameras x num_points x 4
    # points_camera_frame = torch.bmm(w2c, points_h.permute(0,2,1)).permute(0,2,1)
    points_camera_frame = torch.einsum('bik,nk->bni', w2c, points_h)
    
    # Remove the homogeneous coordinate
    points_camera_frame = points_camera_frame[:, :, :3]

    mask_z = (-points_camera_frame[:, :, 2] > z_min) & (-points_camera_frame[:, :, 2] < z_max)
    
    # points_cam = points_cam_h[:, :, :] / points_cam_h[:, :, [2]]  # Normalize by Z
    
    # Check Z value
    # breakpoint()
    uv_h = torch.einsum('bik,bnk->bni', K, points_camera_frame)  # num_cameras x num_points x 3

    uv = uv_h/uv_h[:, :, [2]]

    u = uv[:, :, 0]
    v = uv[:, :, 1]
    
    # Project points to image plane
    # u = fx[:, None] * points_cam[:, :, 0] / points_cam[:, :, 2]
    # v = fy[:, None] * points_cam[:, :, 1] / points_cam[:, :, 2]
    
    # Check if pixel coordinates are within image boundaries
    mask_u = (u >= 0) & (u < width)
    mask_v = (v >= 0) & (v < height)
    mask_uv = mask_u & mask_v
    
    # Combine masks
    visibility = (mask_z & mask_uv)
    # breakpoint()
    return visibility

def get_visibility(points, cameras, model, threshold = 0.01):
    torch.cuda.empty_cache()
    original_shape = points.shape
    points = points.view(-1,3)

    in_fov = fov_visibility(cameras, points)

    n_cameras = cameras.camera_to_worlds.shape[0]
    n_points = points.shape[0]

    opacity = torch.ones(n_points, dtype=torch.float32).cuda()

    

    for i in range(n_cameras):
        # breakpoint()
        compute_points_mask = (in_fov[i,...] & (opacity>threshold))
        if compute_points_mask.any():
            compute_points = points[compute_points_mask,:]
            if use_uniform:
                opacity[compute_points_mask] *= single_camera_field_opacity(compute_points, cameras.camera_to_worlds[i,...], model)
            else:
                opacity[compute_points_mask] *= single_camera_field_opacity_adaptive_sampler(compute_points, cameras.camera_to_worlds[i,...], model)

    opacity = opacity.view(*original_shape[:-1],1)
    return 1.0 - opacity



@torch.no_grad()
def get_density(points, model):
    directions = torch.zeros_like(points)
    pixel_area = torch.zeros(points.shape[0], 1).to(points)
    starts = torch.zeros(points.shape[0], 1).to(points)
    ends = torch.ones(points.shape[0], 1).to(points)
    frustums = Frustums(origins=points, directions=directions, starts=starts, ends=ends, pixel_area=pixel_area)
    camera_indices = torch.zeros(points.shape[0], 1, dtype=int, device=points.device)
    ray_samples = RaySamples(frustums, camera_indices=camera_indices)
    density = model.field(ray_samples)
    return density[FieldHeadNames.DENSITY]

@torch.no_grad()    
def get_density_embedding(points, model):
    directions = torch.zeros_like(points)
    pixel_area = torch.zeros(points.shape[0], 1).to(points)
    starts = torch.zeros(points.shape[0], 1).to(points)
    ends = torch.ones(points.shape[0], 1).to(points)
    frustums = Frustums(origins=points, directions=directions, starts=starts, ends=ends, pixel_area=pixel_area)
    camera_indices = torch.zeros(points.shape[0], 1, dtype=int)
    ray_samples = RaySamples(frustums, camera_indices=camera_indices)
    _, density_embedding = model.field.get_density(ray_samples)
    return density_embedding
