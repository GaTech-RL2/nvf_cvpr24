import torch
from PIL import Image, ImageDraw, ImageFont
from nvf.env.utils import get_images, GIFSaver, stack_img, save_img, empty_cache
from nerfstudio.cameras.cameras import Cameras, CameraType
import pickle as pkl
from nvf.active_mapping.active_mapping import ActiveMapper, get_entropy_for_camera_ray_bundle
from nvf.nerfstudio_interface.nbv_dataparser import NBVDataParserConfig
from matplotlib import pyplot as plt
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
import matplotlib.cm as cm

from nerfstudio.exporter.exporter_utils import generate_point_cloud
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree as KDTree
from scipy.spatial.transform import Rotation as R
from typing import Optional

from nvf.metric.mesh_metrics import VisibilityMeshMetric, GeometryMeshMetric

class RefMetricTracker(object):
    debug = False
    """docstring for RefMetric"""
    # fov = np.pi/3
    tb_writer: Optional[SummaryWriter] = None
    base_log_dir = 'results/log'


    def __init__(self, config=None, env=None):
        super(RefMetricTracker, self).__init__()
        self.config = config
        self.env = env
        if not env is None:
            self.mesh = env.scene.mesh
            np_images, transforms = self.env.get_images(mode='eval',return_quat=False)
        else:
            raise ValueError('env is None')
            # np_images, transforms = get_images(file='data/nerfstudio/hubble_reference/transforms.json', return_quat=False)
        self.ref_images = torch.stack([torch.FloatTensor(iii)[...,:3].permute(2,0,1) for iii in np_images])
        if self.ref_images.max()>1.1:
            self.ref_images /= 255.
        self.ref_poses = torch.stack(transforms)
        
        self.gif = GIFSaver()
        self.img_size = np_images[0].shape[:-1]
        self.i=0
        # self.loss_hist = []

        self.pred_img_hist = []
        self.entropy_hist = []

        self.metric_fn = {
            'psnr': None,#PeakSignalNoiseRatio(data_range=1.0),
            'ssim': None,#, structural_similarity_index_measure,
            'lpips': None,#, LearnedPerceptualImagePatchSimilarity(normalize=True),
            'rgb_loss': None,#torch.nn.MSELoss(),
        }

        self.geometry_metric = GeometryMeshMetric(scene=env.scene, num_sample_points=500000)
        self.visibility_metric = VisibilityMeshMetric(scene=env.scene)

        self.metric_hist = {k:[] for k in self.metric_fn.keys()}
        self.metric_hist.update({k:[] for k in ['acc','comp','cr','vis','vis_area']})

        # self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        # self.ssim = structural_similarity_index_measure
        # self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        # self.rgb_loss = torch.nn.MSELoss()

        self.trajectory = []
        self.trajectory_start_idx = 0

    
    def setup_writer(self, relative_log_dir):
        writer_log_path = os.path.join(self.base_log_dir, relative_log_dir)
        self.tb_writer = SummaryWriter(log_dir=writer_log_path)

    # def __call__(self,*args, **kwargs):
    #     return self.forward(*args, **kwargs)
    def init_trajectory(self, pose):
        self.trajectory = [pp.detach().cpu().numpy() for pp in pose]
        self.trajectory_start_idx = len(pose)

        for pp in pose:
            self.visibility_metric.update(pp[None,...])
    
    def update_trajectory(self, pose):
        self.trajectory += [pp.detach().cpu().numpy() for pp in pose]

    def get_metric(self, pipeline):
        # output = pipeline.visualize(self.ref_poses, height=H, width=W, render_option='rgb')
        metric_dict = {}
        output = self.visualize(pipeline)
        # import pdb; pdb.set_trace()
        output = output.to(pipeline.trainer.device)
        self.ref_images = self.ref_images.to(output)
        for k,metric in self.metric_fn.items():
            if metric is None:
                # breakpoint()
                metric = getattr(pipeline.trainer.pipeline.model, k)
                # print(metric)
            # breakpoint()
            loss = metric(output.clip(0.,1.), self.ref_images.clip(0.,1.))
            # self.metric_hist[k].append(loss.mean().item())
            metric_dict[k] = loss.mean().item()

            del loss
            empty_cache()

        

        # for pp in pose:
        #     self.visibility_metric.update(pp[None,...])
        
        accuracy, completion, completion_ratio = self.geometry_metric.compute_pcl_metric(pipeline.trainer.pipeline)
        visibility, vis_area = self.visibility_metric.get_value()
        empty_cache()

        for k,value in zip(['acc','comp','cr','vis','vis_area'], [accuracy, completion, completion_ratio, visibility, vis_area]):
            # if self.tb_writer is not None:
            #     self.tb_writer.add_scalar(f'eval/{k}', value, step)
            # self.metric_hist[k].append(value)
            metric_dict[k] = value
        
        return metric_dict, output

        

    def update(self, pipeline, pose, step):
        empty_cache()
        self.update_trajectory(pose)
        # import pdb; pdb.set_trace()
        # self.ref_images = self.ref_images.to(pipeline.trainer.pipeline.device)
        # self.ref_poses = self.ref_poses.to(pipeline.trainer.pipeline.device)
        H,W = self.img_size
        with torch.no_grad():
            
            metric_dict, output = self.get_metric(pipeline)

            for k,v in metric_dict.items():
                self.metric_hist[k].append(v)
                self.tb_writer.add_scalar(f'eval/{k}', v, step)
            
            print('eval', {k:round(self.metric_hist[k][-1],4) for k in ['psnr', 'rgb_loss', 'acc', 'comp','cr','vis', 'vis_area']})
            for pp in pose:
                self.visibility_metric.update(pp[None,...])

            if self.debug:
                fig, (ax1, ax2) = plt.subplots(1,2)

                ax1.imshow(output[3].permute(1,2,0).cpu().numpy())
                ax2.imshow(self.ref_images[3].permute(1,2,0).cpu().numpy())
                plt.pause(0.1)
                breakpoint()
        # self.loss_hist.append(loss.sum().item())
        self.pred_img_hist.append(output.detach().cpu().numpy())

        entropy = self.get_entropy(pipeline)

        self.entropy_hist.append(entropy.detach().cpu().numpy())

        images = output.permute(0,2,3,1).detach().cpu().numpy()

        images = images.reshape(5,-1,H,W,3)

        
        stacked_images = stack_img(images,sep=[10,10])
        # save_img(stacked_images, 'results/temp.png')
        

        fn=lambda img: ImageDraw.Draw(img).text((3, 3), f"T={self.i}", fill=(255, 255, 255))
        self.gif.add(stacked_images, fn=fn)

        self.write_scatter(self.config.camera_aabb.T)
        self.i+=1

    @torch.no_grad()
    def visualize(self, pipeline):
        n_cameras = self.ref_poses.shape[0]

        fov = self.config.env.fov /180 *np.pi
        width = self.img_size[1] * torch.ones(n_cameras,1, dtype=torch.float32)
        height = self.img_size[0] * torch.ones(n_cameras,1, dtype=torch.float32)
        fx = 0.5*width/np.tan(fov/2)
        # fx = torch.ones(n_cameras,1, dtype=torch.float32)*1024
        fy = fx
        
        cx = width//2
        cy = height//2
        # breakpoint()
        all_images = []
        for i in range(n_cameras):
            cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            camera_type=CameraType.PERSPECTIVE,
            camera_to_worlds=self.ref_poses[i,:-1,:]
            ).to(pipeline.trainer.device)

            camera_ray_bundle = cameras.generate_rays(camera_indices=0, aabb_box=None)
            # print('gen')
            outputs = pipeline.trainer.pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
            # print('rendered')
            image = outputs["rgb"]
            all_images.append(image)
        all_images = torch.moveaxis(torch.stack(all_images), -1, 1)
        # print(all_images.shape)

        return all_images
    
    @torch.no_grad()
    def get_entropy(self, pipeline):
        if hasattr(pipeline.trainer.pipeline.model.renderer_entropy, 'depth_threshold'):
            original_depth_threshold = pipeline.trainer.pipeline.model.renderer_entropy.depth_threshold
        else:
            original_depth_threshold = None
        n_cameras = self.ref_poses.shape[0]

        fov = self.config.env.fov /180 *np.pi
        width = self.img_size[1] * torch.ones(n_cameras,1, dtype=torch.float32)
        height = self.img_size[0] * torch.ones(n_cameras,1, dtype=torch.float32)
        fx = 0.5*width/np.tan(fov/2)
        # fx = torch.ones(n_cameras,1, dtype=torch.float32)*1024
        fy = fx
        
        cx = width//2
        cy = height//2
        # all_images = []
        bs = self.ref_poses.shape[0]
        costs = torch.zeros(bs, self.img_size[0], self.img_size[1])
        for i in range(n_cameras):
            cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            camera_type=CameraType.PERSPECTIVE,
            camera_to_worlds=self.ref_poses[i,:-1,:]
            ).to(pipeline.trainer.device)

            camera_ray_bundle = cameras.generate_rays(camera_indices=0, aabb_box=None)
            # print('gen')
            entropy = get_entropy_for_camera_ray_bundle(pipeline.trainer.pipeline.model, camera_ray_bundle)
            # print('rendered')
            costs[i] = entropy.squeeze(-1)
           
        # print(all_images.shape)
        if original_depth_threshold is not None:
            pipeline.trainer.pipeline.model.renderer_entropy.depth_threshold = original_depth_threshold
        return costs

    def write_scatter(self, aabb):
        points = np.array(self.trajectory)[...,-3:].reshape(-1,3)
        start_idx = self.trajectory_start_idx

        # fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        fig = plt.figure()
        ax = plt.gca()
    
        # Create color arrays
        colors_before_start = np.array([[0, 0, 0, 1]] * start_idx)  # Black with full opacity
        colormap_indices = (np.arange(start_idx, len(points)) - start_idx) / (len(points) - start_idx)
        colors_after_start = cm.autumn(colormap_indices)
        colors = np.vstack([colors_before_start, colors_after_start])
        
        
        # XY plane
        ax.clear()
        ax.scatter(points[:, 0], points[:, 1], color=colors)
        ax.set_xlim(aabb[0])
        ax.set_ylim(aabb[1])
        ax.set_title('XY Plane')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        self.tb_writer.add_figure('Pose_XY', fig, self.i)
        
        # YZ plane
        ax.clear()
        ax.scatter(points[:, 1], points[:, 2], color=colors)
        ax.set_xlim(aabb[1])
        ax.set_ylim(aabb[2])
        ax.set_title('YZ Plane')
        ax.set_xlabel('Y')
        ax.set_ylabel('Z')
        self.tb_writer.add_figure('Pose_YZ', fig, self.i)

        # ZX plane
        ax.clear()
        ax.scatter(points[:, 0], points[:, 2], color=colors)
        ax.set_xlim(aabb[0])
        ax.set_ylim(aabb[2])
        ax.set_title('XZ Plane')
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        self.tb_writer.add_figure('Pose_XZ', fig, self.i)
        # breakpoint()
        # plt.tight_layout()

    def save_metric(self, file):
        result = {
            'metric':self.metric_hist,
            'trajectory':self.trajectory,
            'trajectory_start_idx ':self.trajectory_start_idx,
        }
        with open(file, 'wb') as f:
            pkl.dump(result, f)