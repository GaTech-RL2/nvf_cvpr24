
import torch
from PIL import Image, ImageDraw, ImageFont
from nvf.env.utils import get_images, GIFSaver, stack_img, save_img, empty_cache
from nerfstudio.cameras.cameras import Cameras, CameraType
import pickle as pkl
from nvf.active_mapping.active_mapping import ActiveMapper
from nvf.active_mapping.mapping_utils import to_transform
from nvf.nerfstudio_interface.nbv_dataparser import NBVDataParserConfig
from matplotlib import pyplot as plt
import numpy as np

from nerfstudio.exporter.exporter_utils import generate_point_cloud
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree as KDTree
from scipy.spatial.transform import Rotation as R

from pytorch3d.structures import Meshes
from pytorch3d.renderer.cameras import PerspectiveCameras, FoVPerspectiveCameras
from pytorch3d.renderer import RasterizationSettings, MeshRasterizer, MeshRenderer, SoftSilhouetteShader, SoftPhongShader, PointLights, TexturesVertex, HardPhongShader, look_at_view_transform
from pytorch3d.transforms import quaternion_to_matrix

from pytorch3d.renderer.mesh.textures import Textures
from pytorch3d.renderer.mesh.shader import TexturedSoftPhongShader

from pytorch3d.renderer.blending import BlendParams

from pytorch3d.io import IO, load_objs_as_meshes

from nvf.env.Scene import *
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional

from nvf.env.utils import pose_point_to, pose2tensor, tensor2pose

from nvf.env.utils import get_conf, pose_point_to, rgb_to_rgba

from gtsam import Pose3

import time

class FaceIndexShader(torch.nn.Module):
    def forward(self, fragments, meshes, **kwargs):
        return fragments.pix_to_face

class VisibilityMeshMetric(object):
    
    def __init__(
        self, 
        scene,
        mesh_path = None,
    ):
        super(VisibilityMeshMetric, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.fx = scene.fx
        self.fy = scene.fy
        self.cx = scene.height / 2.0
        self.cy = scene.width / 2.0

        self.H = scene.height
        self.W = scene.width

        if mesh_path == None:
            mesh = scene.mesh
            vertices = torch.tensor(mesh.vertices, dtype=torch.float32).to(self.device)
            faces = torch.tensor(mesh.faces, dtype=torch.int64).to(self.device)

            constant_color = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
            vertex_colors = constant_color[None, None, :].repeat(1, vertices.shape[0], 1)
            textures = TexturesVertex(verts_features=vertex_colors)

            self.mesh = Meshes(verts=[vertices], faces=[faces], textures=textures)

            if scene.name in ['hotdog', 'room', 'materials', 'ship']:
                self.raster_settings = RasterizationSettings(image_size=(self.H, self.W), 
                                                            blur_radius=0.0,
                                                            bin_size=0,
                                                            faces_per_pixel=1,)
            else:
                self.raster_settings = RasterizationSettings(image_size=(self.H, self.W), 
                                                    blur_radius=0.0,
                                                    faces_per_pixel=1)
        else:
            mesh = o3d.io.read_triangle_mesh(mesh_path)

            num_sample_points = len(mesh.vertices)

            vertices = torch.tensor(mesh.vertices, dtype=torch.float32).to(self.device)
            faces = torch.tensor(mesh.triangles, dtype=torch.int64).to(self.device)

            constant_color = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
            vertex_colors = constant_color[None, None, :].repeat(1, vertices.shape[0], 1)
            textures = TexturesVertex(verts_features=vertex_colors)

            self.mesh = Meshes(verts=[vertices], faces=[faces], textures=textures)
        
            if scene.name in ['hotdog', 'room', 'materials', 'ship']:
                self.raster_settings = RasterizationSettings(image_size=(self.H, self.W), 
                                                            blur_radius=0.0,
                                                            bin_size=0,
                                                            faces_per_pixel=1,)
            else:
                self.raster_settings = RasterizationSettings(image_size=(self.H, self.W), 
                                                            blur_radius=0.0,
                                                            faces_per_pixel=1,
                                                            max_faces_per_bin=num_sample_points + 50)


        self.shader = FaceIndexShader()

        n_faces = self.mesh.faces_packed().shape[0]
        self.face_visibility = torch.zeros(n_faces, dtype=torch.float32, device=self.device)

    def update(self, poses):
        """
        args:
            pose: Float[Tensor, "*batch 7"]
        # """

        n, _ = poses.shape

        for i in range(n):
            pose = poses[i]
            transform = tensor2pose(pose).matrix()
            transform = torch.from_numpy(transform).to(self.device)
            transform = transform.type(torch.float32)

            R_ = transform[:3,:3]
            t = transform[:3,3]

            R_ = (R_ @ torch.diag(torch.tensor([-1.,1., -1.])).to(self.device) )

            t = -R_.t() @ t

            t = t[None,...]
            R_ = R_[None, ...]

            focal_length = torch.tensor((self.fx, self.fy), dtype=torch.float32,).unsqueeze(0).to(self.device)  
            principal_point = torch.tensor((self.cx, self.cy), dtype=torch.float32).unsqueeze(0).to(self.device)
            image_size = torch.tensor((self.H, self.W), dtype=torch.float32).unsqueeze(0).to(self.device)

            cameras = PerspectiveCameras(R=R_, T=t, focal_length=focal_length, principal_point=principal_point, image_size=image_size, in_ndc=False).to(self.device)
            # self.cameras = FoVPerspectiveCameras(R=R_, T=t, fov=self.scene.hfov/np.pi*180)

            rasterizer = MeshRasterizer(cameras, raster_settings=self.raster_settings)
            renderer = MeshRenderer(rasterizer=rasterizer, shader=self.shader)

            face_indices_image = renderer(self.mesh)
            test_image = torch.where(face_indices_image == -1, 0, 255)

            visible_faces = torch.unique(face_indices_image[face_indices_image != -1])
            # print(visible_faces.size())
            self.face_visibility[visible_faces] = 1.0

        
    def get_renders(self, poses):
        try:
            n, _ = poses.shape

        except:
            n = len(poses)

        start = time.time()

        costs = torch.zeros(n, dtype=torch.float32, device=self.device)
        renders = torch.zeros((n, self.H, self.W), dtype=torch.float32, device=self.device)

        for i in range(n):
            pose = poses[i].detach().clone()
            transform = tensor2pose(pose).matrix()
            transform = torch.from_numpy(transform).to(self.device)
            transform = transform.type(torch.float32)

            R_ = transform[:3,:3]
            t = transform[:3,3]

            R_ = (R_ @ torch.diag(torch.tensor([-1.,1., -1.])).to(self.device) )

            t = -R_.t() @ t

            t = t[None,...]
            R_ = R_[None, ...]

            focal_length = torch.tensor((self.fx, self.fy), dtype=torch.float32,).unsqueeze(0).to(self.device)  
            principal_point = torch.tensor((self.cx, self.cy), dtype=torch.float32).unsqueeze(0).to(self.device)
            image_size = torch.tensor((self.H, self.W), dtype=torch.float32).unsqueeze(0).to(self.device)

            cameras = PerspectiveCameras(R=R_, T=t, focal_length=focal_length, principal_point=principal_point, image_size=image_size, in_ndc=False).to(self.device)
            # self.cameras = FoVPerspectiveCameras(R=R_, T=t, fov=self.scene.hfov/np.pi*180)

            rasterizer = MeshRasterizer(cameras, raster_settings=self.raster_settings)
            renderer = MeshRenderer(rasterizer=rasterizer, shader=self.shader)

            face_indices_image = renderer(self.mesh) 

            # image render = only previously unseen faces of the mesh
            # multiple pixels can correspond to same face
            visible_faces = torch.unique(face_indices_image[face_indices_image != -1])
            temp = self.face_visibility.detach().clone()
            temp[visible_faces] = 1.0
            renders[i] = temp[face_indices_image.squeeze()] - self.face_visibility[face_indices_image.squeeze()]

            # sum of visible faces minus previously seen faces
            costs[i] = visible_faces.size(dim=0) - torch.sum(self.face_visibility[visible_faces])

            # del temp

        end = time.time()
        print("time duration: ", end - start)
            
        return renders, costs


    def get_value(self):
        visibility_area = (self.face_visibility * self.mesh.faces_areas_packed()).sum()/self.mesh.faces_areas_packed().sum()
        return self.face_visibility.mean().item(), visibility_area.item()

    def pipeline_preprocessing(self, pipeline, images, poses):
        transform_poses = []
        normalize_images = []
        for i in range(len(poses)):
            transform = to_transform(poses[i]).type(torch.float32)

            transform_poses.append(transform)
            normalize_images.append(images[i].astype("float32") / 255.0)


        pipeline.datamanager.train_image_dataloader.add_data(normalize_images, transform_poses, True)
        pipeline.train()

class GeometryMeshMetric(object):
    """
    Following iMAP metrics: accuracy, completion, completion ratio (Sucar et al 2021 https://arxiv.org/abs/2103.12352), 
    implementation based on https://github.com/cvg/nice-slam/blob/master/src/tools/eval_recon.py
    """
    def __init__(
        self,
        scene,
        num_sample_points: int = 500000,
        dist_threshold: float = 0.01,
    ):
        super(GeometryMeshMetric, self).__init__()

        self.sample_points = num_sample_points
        if scene.name == 'room':
            self.dist_threshold = 0.1
        else:
            self.dist_threshold = dist_threshold
        self.gt_mesh = scene.mesh
        self.nerf_pcd = None

    def compute_pcl_metric(
        self, 
        pipeline, 
        load_ckpt = False, 
        images = None, 
        transforms = None
    ):
        """ Construct point cloud from NeRF model and computes iMAP metrics

        Args:
            pipeline: nbv pipeline class instance
            load_ckpt: flag, if true, populates pipeline with associated images & poses to build the point cloud (necessary if using pipeline loaded from a checkpoint)
            images: list of numpy images based on get_images() output
            transforms: list of torch poses based on get_images output 
        """
            
        if load_ckpt:
            assert images != None
            assert transforms != None
            transform_poses = []
            normalize_images = []
            for i in range(len(transforms)):
                transform = to_transform(transforms[i]).type(torch.float32)

                transform_poses.append(transform)
                normalize_images.append(images[i].astype("float32") / 255.0)

            pipeline.datamanager.train_image_dataloader.add_data(normalize_images, transform_poses, True)
            pipeline.train()

        return self._compute_pcl_metric(pipeline)

    def _compute_pcl_metric(self, pipeline):
        empty_cache()
        temp = pipeline.datamanager.train_pixel_sampler.num_rays_per_batch
        # Increase the batchsize to speed up the evaluation.
        # assert isinstance(pipeline.datamanager, (VanillaDataManager, ParallelDataManager))
        # assert pipeline.datamanager.train_pixel_sampler is not None
        pipeline.datamanager.train_pixel_sampler.num_rays_per_batch = 32768 // 2

        self.nerf_pcd = generate_point_cloud(
            pipeline=pipeline,
            num_points=self.sample_points,
            remove_outliers=True,
            reorient_normals=True,
            estimate_normals="open3d",
            rgb_output_name="rgb",
            depth_output_name="depth",
            normal_output_name= None,
            use_bounding_box=False,
            bounding_box_min=None,
            bounding_box_max=None,
            crop_obb=None,
            std_ratio=10.0,
        )

        pipeline.datamanager.train_pixel_sampler.num_rays_per_batch = temp
        empty_cache()

        return self.get_outputs()

    def get_outputs(self):

        assert self.nerf_pcd != None

        gt_points, _ = trimesh.sample.sample_surface(self.gt_mesh, self.sample_points)
        gt_points = trimesh.PointCloud(vertices=gt_points).vertices

        result = trimesh.exchange.ply.export_ply(trimesh.PointCloud(vertices=gt_points), encoding='ascii')
        nerf_points = self.nerf_pcd.points

        gt_tree = KDTree(gt_points)
        dists, _ = gt_tree.query(nerf_points)
        accuracy = np.mean(dists)

        nerf_tree = KDTree(nerf_points)
        dists, _ = nerf_tree.query(gt_points)
        completion = np.mean(dists)
        
        completion_ratio = np.mean((dists < self.dist_threshold).astype(np.float32))

        return accuracy, completion, completion_ratio