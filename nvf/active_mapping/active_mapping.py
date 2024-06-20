
import glob
import json
import os
import pathlib
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional
from PIL import Image
import PIL 
import gc
import mediapy as media
import numpy as np
import PIL
import torch
import yaml
import time
from jaxtyping import Float, Int
from PIL import Image
from scipy.spatial.transform import Rotation
from torch import Tensor, nn

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.cameras.camera_paths import (get_interpolated_camera_path,
                                             get_path_from_json,
                                             get_camera,
                                             get_spiral_path)
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.base_datamanager import \
    VanillaDataManagerConfig
from nerfstudio.data.dataparsers.instant_ngp_dataparser import \
    InstantNGPDataParserConfig
from nerfstudio.engine.callbacks import (TrainingCallback,
                                         TrainingCallbackAttributes,
                                         TrainingCallbackLocation)
from nerfstudio.engine.optimizers import (AdamOptimizerConfig,
                                          RAdamOptimizerConfig)
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import Trainer, TrainerConfig
from nerfstudio.models.instant_ngp import InstantNGPModelConfig
from nerfstudio.models.nerfacto import NerfactoModelConfig
from nvf.nerfstudio_interface.nbv_datamanager import NBVDataManagerConfig
from nvf.nerfstudio_interface.nbv_dataparser import NBVDataParserConfig

from nvf.nerfstudio_interface.nbv_trainer import NBVTrainer, NBVTrainerConfig
from nvf.nerfstudio_interface.nbv_utils import load_checkpoint

from nerfstudio.pipelines.base_pipeline import Pipeline, VanillaPipelineConfig
from nerfstudio.pipelines.dynamic_batch import DynamicBatchPipelineConfig
from nerfstudio.utils import colormaps, install_checks
from nerfstudio.utils.eval_utils import eval_load_checkpoint, eval_setup
from nerfstudio.utils.io import load_from_json

from nvf.nerfstudio_interface.nbv_utils import load_checkpoint, save_init_checkpoint, load_init_checkpoint

from torch.cuda.amp.grad_scaler import GradScaler

from nvf.active_mapping.mapping_utils import *

class ActiveMapper:

    config: TrainerConfig
    trainer: Trainer
    config_pth: str
    dataset_pth: str
    init_train: bool = True
    model: Literal["instant-ngp", "nerfacto"] = "instant-ngp"
    use_visibility : bool = False
    use_tensorboard : bool = False
    plan_img_size = (256,256)
    train_img_size = (512,512)
    fov = 90
    # scene_scale = 0.3333

    def __init__(self, config_pth: str = "", dataset_pth: str = "") -> None:
        """
        config_pth: string path to a nerfstudio trainer config.yml

        To generate a config.yml with the correct parameters, enter in the terminal (note that you may want a different data path):
            ns-train instant-ngp --data data/nerfstudio/hubble_mask  --pipeline.model.background-color "random" instant-ngp-data
        
        The path to the config.yml file will be show in the terminal, but this doesn't automatically link to the model checkpoint.
        To reference the model checkpoint, pass the model path (in a json pathlib.Path format) into the config parameter: load_dir

        """
        if len(config_pth) > 0:
            self.config_pth = config_pth
            self.config = yaml.load(pathlib.Path(config_pth).absolute().read_text(), Loader=yaml.Loader)
        if len(dataset_pth) > 0:
            self.dataset_pth = dataset_pth

    def initialize_config(self, config_home: str, dataset_path: str, model: Literal["instant-ngp", "nerfacto"] = "instant-ngp") -> str:
        """
        Creates a new config file, requiring a given dataset path and a directory path to put the config in

        Args:
            config_home: string path to location config will be stored in
            dataset_path: string path to location of dataset
            model: nerf model for training 
        Returns:
            Outputs the config.yml string file path
        """
        self.config_home = Path(config_home)
        self.model = model
        if model == "instant-ngp":
            config = self.init_instant_ngp_config(config_home=config_home, dataset_path=dataset_path)
            init_step = 30
        else:
            config = self.init_nerfacto_config(config_home=config_home, dataset_path=dataset_path)
            init_step = 30

        self.trainer = self.config.setup(local_rank=0, world_size=1, init_step = init_step)
        self.trainer.setup()
        self.trainer.config_home = self.config_home

        self.trainer.config.logging.steps_per_log = 10

        return config

    def load_ckpt_path(self, checkpoint_path: str) -> None:
        assert len(self.config_pth) > 0
        assert self.config != None

        # self.config['load_dir'] = Path(checkpoint_path_dir)
        self.config['load_file'] = Path(checkpoint_path)

        path = Path(self.config_pth)
        path.write_text(yaml.dump(self.config))  
    
    def load_ckpt(self, path: str) -> None:

        load_checkpoint(Path(path), self.trainer.pipeline)

    def save_ckpt(self, filepath, step=0) -> None:
        """Save the model and optimizers

        Args:
            step: number of steps in training for given checkpoint
        """
        # possibly make the checkpoint directory
        path = os.path.dirname(filepath)
        if not os.path.exists(path):
            os.makedirs(path)
        # save the checkpoint
        torch.save(
            {
                "step": step,
                "pipeline": self.trainer.pipeline.module.state_dict()  # type: ignore
                if hasattr(self.trainer.pipeline, "module")
                else self.trainer.pipeline.state_dict(),
                "optimizers": {k: v.state_dict() for (k, v) in self.trainer.optimizers.optimizers.items()},
                "scalers": self.trainer.grad_scaler.state_dict(),
            },
            filepath,
        )

    def use_training_ckpt(self, toggle: bool) -> None:
        """
        enable/disable using the trainer config's model checkpoint
        """
        assert len(self.config_pth) > 0
        assert self.config != None

        if toggle == True:
            self.config['load_dir'] = Path(str(Path(self.config_pth).resolve().parents[0]) + "/dataset/" + self.model +"/latest_checkpoint/nerfstudio_models")
        else:
            self.config['load_dir'] = None
        
        path = Path(self.config_pth)
        path.write_text(yaml.dump(self.config))     

    def add_image(
        self,
        images: List[NDArray] = [],
        poses: List = [],
        model_option: Literal["reinit", "load_ckpt", None] = None
    ) -> None:
        """
        adds new images to datamananager, begins training loop from last checkpoint (assuming toggle_config_model_checkpoint(True) has been called)


        Args:
            images: list of numpy images of dim (H, W, C), where C=4 for RGBA, and assumes intensity values are between 0-255
            poses: list of 1D numpy pose=[quaternion, position]
            load_ckpt: if true, load in a checkpoint before training model
        """

        assert len(self.dataset_pth) > 0

        if model_option == "load_ckpt":
            del self.trainer.pipeline._model
            del self.trainer.optimizers

            self.trainer.pipeline._model = self.trainer.pipeline.config.model.setup(
                scene_box=self.trainer.pipeline.datamanager.train_dataset.scene_box,
                num_train_data=len(self.trainer.pipeline.datamanager.train_dataset),
                metadata=self.trainer.pipeline.datamanager.train_dataset.metadata,
                device=self.trainer.device,
                grad_scaler=self.trainer.grad_scaler,
            )
            self.trainer.pipeline.model = self.trainer.pipeline._model
            self.trainer.pipeline._model.to(self.trainer.device)
            self.trainer.pipeline.train()

            self.trainer.optimizers = self.trainer.setup_optimizers()

            if self.config['load_dir'] == None:
                load_checkpoint(self.config['load_file'], self.trainer.pipeline)
            else:
                eval_load_checkpoint(self.config, self.trainer.pipeline)

        # convert numpy poses into transform matrices
        # convert numpy images into correct format
        transform_poses = []
        normalize_images = []
        for i in range(len(poses)):
            transform = to_transform(poses[i])
            transform_poses.append(transform)
            normalize_images.append(images[i].astype("float32") / 255.0)

        self.trainer.update_data(images=normalize_images, poses=transform_poses)

        # use_var = self.trainer.pipeline.model.field.use_rgb_variance
        # use_visibility = self.trainer.pipeline.model.field.use_visibility
        # use_nvf = self.trainer.pipeline.model.use_nvf
        # calculate_entropy = self.trainer.pipeline.model.calculate_entropy

        # self.trainer.pipeline.model.field.use_rgb_variance = False
        # self.trainer.pipeline.model.field.use_visibility = False
        # self.trainer.pipeline.model.use_nvf = False
        # self.trainer.pipeline.model.calculate_entropy = False
        self.trainer.train()

        
        # self.trainer.pipeline.model.field.use_rgb_variance = use_var
        # self.trainer.pipeline.model.use_nvf = use_nvf


        if self.use_visibility:
            assert self.trainer.pipeline.model.field.use_visibility
            self.trainer.train_nvf()
            # self.trainer.save_checkpoint(self.config['max_num_iterations']-1)
        # self.trainer.pipeline.model.calculate_entropy = True
        # self.trainer.pipeline.model.field.use_visibility = use_visibility
    
    def reset(self):
        del self.trainer.pipeline._model
        del self.trainer.optimizers

        self.trainer.init_step = -1
        torch.cuda.empty_cache()
        gc.collect()
        self.trainer.grad_scaler = GradScaler(enabled=self.trainer.use_grad_scaler)

        self.trainer.pipeline._model = self.trainer.pipeline.config.model.setup(
            scene_box=self.trainer.pipeline.datamanager.train_dataset.scene_box,
            num_train_data=len(self.trainer.pipeline.datamanager.train_dataset),
            metadata=self.trainer.pipeline.datamanager.train_dataset.metadata,
            device=self.trainer.device,
            grad_scaler=self.trainer.grad_scaler,
        )

        self.trainer.pipeline._model.to(self.trainer.device)
        self.trainer.pipeline.train()

        self.trainer.optimizers = self.trainer.setup_optimizers()
        
        self.trainer.pipeline._model.update_to_step(0)
        self.trainer._start_step = 0

        if hasattr(self.trainer.pipeline, 'reset'):
            self.trainer.pipeline.reset()
        
        self.trainer.callbacks = self.trainer.pipeline.get_training_callbacks(
            TrainingCallbackAttributes(
                optimizers=self.trainer.optimizers,
                grad_scaler=self.trainer.grad_scaler,
                pipeline=self.trainer.pipeline,
            )
        )
        
        torch.cuda.empty_cache()
        gc.collect()

    def train(self) -> None:
        """
        train NeRF model
        """
        assert len(self.config_pth) > 0
        assert len(self.dataset_pth) > 0
        assert self.config != None

        self.trainer.train()

    def get_cost(
        self, poses: Float[Tensor, "*bs trajectory_length 7"], use_uniform: bool = False, 
        load_ckpt: bool = False,
        return_image: bool = False,
        height: int = 300,
        width: int = 300,
    ) -> Float[Tensor, "*bs trajectory_length 1"]:
        """ 1) convert camera poses tensor to a camera dictionary format
            2) convert camera dictionary to camera object
            3) pass camera object to pipeline, outputs entropy tensor
        """
        opt_agent = True if poses.requires_grad else False
        if load_ckpt:
            assert len(self.config_pth) > 0
            assert self.config != None

            pipeline = self.trainer.config.pipeline.setup(
                device=self.trainer.device,
                test_mode="inference",
            )
            pipeline.eval()

            if self.config['load_dir'] == None and self.config['load_file'] != None:
                load_checkpoint(self.config['load_file'], pipeline)
            else:
                eval_load_checkpoint(self.config, pipeline)
        else:
            pipeline = self.trainer.pipeline
        
        orig_shape = poses.shape[:-1]
        assert poses.shape[-1] == 7
        poses = poses.view(-1, 7)

        bs = poses.shape[0]
        if not return_image:
            costs = torch.zeros(bs, device=poses.device)
        else:
            costs = torch.zeros(bs, height, width, device=poses.device)

        for batch in range(bs):
            if opt_agent:
                transform = quat_to_rot_transform_torch(poses[batch])
                camera = get_camera(transform, height, width, self.fov)
                camera = camera.to(pipeline.device)  
            else:
                transform = to_transform( poses[batch])
                camera_dict = get_camera_dict(transform, self.fov, height, width)
                camera = get_path_from_json(camera_dict)
                camera = camera.to(pipeline.device)

            # camera ray bundle 300x300
            camera_ray_bundle = camera.generate_rays(camera_indices=0, aabb_box=None) 
            
            if opt_agent:
                entropy = get_entropy_for_camera_ray_bundle_diff(pipeline.model, camera_ray_bundle)
            else:
                entropy = get_entropy_for_camera_ray_bundle(pipeline.model, camera_ray_bundle)
            # entropy = pipeline.model.get_entropy(camera_ray_bundle.flatten())
            if not return_image:
                costs[batch] = torch.mean(entropy)
                costs = costs.view(*orig_shape)
            else:
                costs[batch] = entropy.squeeze(-1)
                costs = costs.view(*orig_shape, height, width)

        if load_ckpt:
            del pipeline
        return costs

    def render_image_output(
        self, 
        image: Float[Tensor, "H W C"],
        output_dir: str = ""
    ) -> None:
        """Render image for visualization/debugging purposes"""
        if output_dir == "":
            path = str(Path(self.config_pth).resolve().parents[0])
            if os.path.exists(path + "/render_output") == False:
                os.mkdir(path + "/render_output")
            render_dir = path + "/render_output"
            media.write_image(
            render_dir +"/"+ str(len(os.listdir(render_dir))) + ".jpg", image.cpu().numpy(), fmt="jpeg", quality=95
            )
        else:
            if os.path.exists(output_dir + "/render_output") == False:
                os.mkdir(output_dir + "/render_output")
            media.write_image(
                output_dir+"/render_output/" + str(len( os.listdir(output_dir + "/render_output/") ))+ ".jpg", image.cpu().numpy(), fmt="jpeg", quality=95
            )

    def visualize(
        self, 
        poses: Float[Tensor, "*batch 7"],
        render_option: Literal["rgb", "entropy", "entropy_colormap", "normalized_colormap"] = "rgb",
        height: int = 300,
        width: int = 300,
        create_image_files: bool = False,
        output_dir: str = "", 
        use_uniform: bool = False,
        load_ckpt: bool = False,
        background_color: Literal["random", "white"] = "random"
    ) -> Float[Tensor, "*batch C H W"]:
        """
        render the images at each pose 
        
        Args:
            poses: Tensor of camera poses 4D quaternion, 3D position
            render_option: format option of visualized images
            height: image height
            width: image width
            create_image_files: option to write visualization to image files
            output_dir: location of image files to be written to

        Return:
            tensor of visualize images
        """
        orig_shape = poses.shape[:-1]
        assert poses.shape[-1] == 7
        poses = poses.view(-1, 7)

        if render_option == "entropy":
            channels = 1
        else:
            channels = 3
        batch, _ = poses.shape
        rendered_images = torch.zeros(batch, channels, height, width)

        self.trainer.pipeline.model.renderer_rgb.background_color = background_color

        if load_ckpt:
            assert len(self.config_pth) > 0
            assert self.config != None

            pipeline = self.trainer.config.pipeline.setup(
                device=self.trainer.device,
                test_mode="inference",
            )
            pipeline.eval()

            if self.config['load_dir'] == None:
                load_checkpoint(self.config['load_file'], pipeline)
            else:
                eval_load_checkpoint(self.config, pipeline)
        else:
            pipeline = self.trainer.pipeline

        if render_option != "rgb":
            pipeline.model.populate_entropy_modules()
            # pipeline.model.use_uniform_sampler = use_uniform

        for i in range(batch):
            transform = to_transform(poses[i])
            camera_dict = get_camera_dict(transform=transform, fov=self.fov, height=height, width=width)
            camera = get_path_from_json(camera_dict)
            camera = camera.to(pipeline.device)

            camera_ray_bundle = camera.generate_rays(camera_indices=0, aabb_box=None)
            outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
            if render_option == "rgb":
                image = outputs["rgb"]
                if create_image_files == True:
                    self.render_image_output(image=image, output_dir=output_dir)
                image = image.movedim(-1, 0)
                rendered_images[i] = image
            else:
                image = outputs["entropy"]
                if render_option == "entropy_colormap":
                    image = colormaps.apply_colormap(
                                image=image,
                                colormap_options=colormaps.ColormapOptions(),
                            )
                elif render_option == "normalized_colormap":
                    image = colormaps.apply_colormap(
                                image=image,
                                colormap_options=colormaps.ColormapOptions(normalize=True),
                            )
                if create_image_files == True:
                    self.render_image_output(image=image, output_dir=output_dir)
                image = image.movedim(-1, 0)
                rendered_images[i] = image

        if load_ckpt:
            del pipeline

        self.trainer.pipeline.model.renderer_rgb.background_color = "random"

        return rendered_images.view(*orig_shape, channels, height, width)

    def init_instant_ngp_config(self, config_home: str, dataset_path: str) -> str:
        """
        Creates instant-ngp config yaml

        Args:
            config_home: string path to location config will be stored in
            dataset_path: string path to location of dataset
        Returns:
            Outputs the config.yml string file path
        """
        num_iterations = 10000

        if os.path.exists(config_home + "/config") == False:
            os.mkdir(config_home + "/config")
            os.mkdir(config_home + "/config/dataset")
            os.mkdir(config_home + "/config/dataset/instant-ngp")
            os.mkdir(config_home + "/config/dataset/instant-ngp/latest_checkpoint")
        if os.path.exists(config_home + "/config/dataset") == False:
            os.mkdir(config_home + "/config/dataset")
            os.mkdir(config_home + "/config/dataset/instant-ngp")
            os.mkdir(config_home + "/config/dataset/instant-ngp/latest_checkpoint")

        path = Path(config_home + "/config/config.yml")

        self.config_pth = str(path.absolute())
        self.dataset_pth = dataset_path

        self.config = NBVTrainerConfig()

        self.config['data'] = Path(dataset_path).absolute()
        self.config['experiment_name'] = 'dataset'
        self.config['max_num_iterations'] = num_iterations
        self.config['method_name'] = 'instant-ngp'
        self.config['mixed_precision'] = True
        
        self.config['optimizers']['fields']['optimizer']['eps'] = 1.0e-15
        self.config['optimizers']['fields']['optimizer']['lr'] = 0.01
        self.config['optimizers']['fields']['optimizer']['weight_decay'] = 0

        self.config['optimizers']['fields']['scheduler'] = ExponentialDecaySchedulerConfig()
        self.config['optimizers']['fields']['scheduler']['lr_final'] = 0.0001
        self.config['optimizers']['fields']['scheduler']['lr_pre_warmup'] = 1.0e-08
        self.config['optimizers']['fields']['scheduler']['max_steps'] = num_iterations
        self.config['optimizers']['fields']['scheduler']['ramp'] = 'cosine'
        self.config['optimizers']['fields']['scheduler']['warmup_steps'] = 0

        self.config['output_dir'] = Path(config_home + "/config/") 

        self.config['pipeline'] = DynamicBatchPipelineConfig()
        dataparser = NBVDataParserConfig(fov = self.fov, 
                                        width = self.train_img_size[0], 
                                        height = self.train_img_size[1])
        self.config['pipeline']['datamanager'] = NBVDataManagerConfig(
                                                    dataparser=dataparser,
                                                    train_num_rays_per_batch=4096, #4096
                                                    eval_num_rays_per_batch=4096,
                                                    num_training_images = 401,
                                                )
        self.config['pipeline']['datamanager']['data'] = Path(dataset_path).absolute()
        self.config['pipeline']['datamanager']['dataparser'] = dataparser
        self.config['pipeline']['datamanager']['dataparser']['data'] = Path(dataset_path).absolute()
        self.config['pipeline']['datamanager']['eval_num_rays_per_batch'] = 4096
        self.config['pipeline']['datamanager']['train_num_rays_per_batch'] = 4096

        self.config['pipeline']['model'] = InstantNGPModelConfig()
        self.config['pipeline']['model']['eval_num_rays_per_batch'] = 8192
        self.config['pipeline']['model']['max_res'] = 512
        
        self.config['pipeline']['model']['background_color'] = "random"
        self.config['load_dir'] = None

        self.config['steps_per_save'] = 200
        self.config['timestamp'] = 'latest_checkpoint'
        self.config['viewer'] = ViewerConfig()
        self.config['viewer']['quit_on_train_completion'] = True
        self.config['viewer']['num_rays_per_chunk'] = 4096
        self.config['viewer']['websocket_port'] = 7007
        self.config['vis'] = "viewer+tensorboard" if self.use_tensorboard else 'viewer'
        self.config['logging']['local_writer']['max_log_size'] = 1
        self.config['load_file'] = None

        path.write_text(yaml.dump(self.config))      

        return self.config_pth

    def init_nerfacto_config(self, config_home: str, dataset_path: str) -> str:
        """
        Creates nerfacto config yaml

        Args:
            config_home: string path to location config will be stored in
            dataset_path: string path to location of dataset
        Returns:
            Outputs the config.yml string file path
        """
        num_iterations = 1001

        if os.path.exists(config_home + "/config") == False:
            os.mkdir(config_home + "/config")
            os.mkdir(config_home + "/config/dataset")
            os.mkdir(config_home + "/config/dataset/nerfacto")
            os.mkdir(config_home + "/config/dataset/nerfacto/latest_checkpoint")
        if os.path.exists(config_home + "/config/dataset/") == False:
            os.mkdir(config_home + "/config/dataset")
            os.mkdir(config_home + "/config/dataset/nerfacto")
            os.mkdir(config_home + "/config/dataset/nerfacto/latest_checkpoint")
        if os.path.exists(config_home + "/config/dataset/nerfacto") == False:
            os.mkdir(config_home + "/config/dataset/nerfacto")
            os.mkdir(config_home + "/config/dataset/nerfacto/latest_checkpoint")

        path = Path(config_home + "/config/config.yml")

        self.config_pth = str(path.absolute())
        self.dataset_pth = dataset_path

        self.config = NBVTrainerConfig(
                        method_name="nerfacto",
                        steps_per_eval_batch=500,
                        steps_per_save=200,
                        max_num_iterations=num_iterations,
                        mixed_precision=True,
                        pipeline=VanillaPipelineConfig(
                            datamanager=NBVDataManagerConfig(
                                dataparser=NBVDataParserConfig(fov = self.fov,height=self.train_img_size[0], width=self.train_img_size[1]),
                                train_num_rays_per_batch=4096,
                                eval_num_rays_per_batch=4096,
                            ),
                            model=NerfactoModelConfig(eval_num_rays_per_chunk=1 << 15),
                        ),
                        optimizers={
                            "proposal_networks": {
                                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
                            },
                            "fields": {
                                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
                            },
                        },
                        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
                    )
        self.config['load_dir'] = None
        self.config['pipeline']['datamanager']['dataparser']['data'] = Path(dataset_path).absolute()

        self.config['data'] = Path(dataset_path).absolute()
        self.config['experiment_name'] = 'dataset'
        self.config['max_num_iterations'] = num_iterations
        self.config['method_name'] = 'nerfacto'
        self.config['mixed_precision'] = True
        
        self.config['optimizers']['proposal_networks']['scheduler'] = ExponentialDecaySchedulerConfig()
        self.config['optimizers']['fields']['optimizer']['eps'] = 1.0e-15
        self.config['optimizers']['fields']['optimizer']['lr'] = 0.01
        self.config['optimizers']['fields']['optimizer']['weight_decay'] = 0

        self.config['optimizers']['fields']['scheduler'] = ExponentialDecaySchedulerConfig()
        self.config['optimizers']['fields']['scheduler']['lr_final'] = 0.0001
        self.config['optimizers']['fields']['scheduler']['lr_pre_warmup'] = 1.0e-08
        self.config['optimizers']['fields']['scheduler']['max_steps'] = num_iterations
        self.config['optimizers']['fields']['scheduler']['ramp'] = 'cosine'
        self.config['optimizers']['fields']['scheduler']['warmup_steps'] = 0

        self.config['output_dir'] = Path(config_home + "/config/") 
        self.config['pipeline']['model']['eval_num_rays_per_batch'] = 8192
        self.config['pipeline']['model']['max_res'] = 512
        self.config['pipeline']['model']['background_color'] = 'random'
        
        self.config['steps_per_save'] = 200
        self.config['timestamp'] = 'latest_checkpoint'
        self.config['viewer'] = ViewerConfig()
        self.config['viewer']['quit_on_train_completion'] = True
        self.config['viewer']['num_rays_per_chunk'] = 4096
        self.config['viewer']['websocket_port'] = 7007
        self.config['vis'] = "viewer+tensorboard" if self.use_tensorboard else 'viewer'
        self.config['logging']['local_writer']['max_log_size'] = 1
        self.config['load_file'] = None

        path.write_text(yaml.dump(self.config))      

        return self.config_pth