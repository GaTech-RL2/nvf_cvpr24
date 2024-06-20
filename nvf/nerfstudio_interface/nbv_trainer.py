from __future__ import annotations

import dataclasses
import json
import time
import glob
import os
# from pathlib import Path
from pathlib import Path

import torch
import torch.nn as nn

from dataclasses import dataclass, field
from typing import Type
from typing_extensions import Literal
import functools

from rich import box, style
from rich.panel import Panel
from rich.table import Table
from rich.console import Console

from nerfstudio.utils.decorators import check_viewer_enabled
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.trainer import Trainer, TrainerConfig

from nerfstudio.viewer.server.viewer_state import ViewerState

from nerfstudio.utils.misc import step_check
from nerfstudio.utils.writer import EventName, TimeWriter
from nerfstudio.utils import profiler, writer
from nerfstudio.utils.rich_utils import CONSOLE

from nerfstudio.fields.nerfacto_field import NerfactoField

from nvf.nerfstudio_interface.nbv_dataset import NBVDataset

from nvf.visibility.train_nvf import gen_data, get_balance_weight, get_density_embedding
from nvf.nerfstudio_interface.nbv_utils import save_init_checkpoint


CONSOLE = Console(width=120)

@dataclass
class NBVTrainerConfig(TrainerConfig):
    _target: Type = field(default_factory=lambda: NBVTrainer)
    num_imgs_to_start: int = 3
    """ Number of images that must be added before training can start. """
    draw_training_images: bool = False
    """ Whether or not to draw the training images in the viewer. """

    nvf_batch_size = 65536
    nvf_num_iterations = 500
    nvf_train_batch_repeat = 5

    """Below methods to allow yaml assignment"""
    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

class NBVTrainer(Trainer):
    config: NBVTrainerConfig
    dataset: NBVDataset
    config_home: os.PathLike
    init_step = 30
    base_log_dir = 'results/log'
    nvf_use_var = False
    nvf_use_var_factor = 0.01

    def __init__(
        self, config: NBVTrainerConfig, local_rank: int = 0, world_size: int = 0, init_step=30
    ):
        # We'll see if this throws and error (it expects a different config type)
        super().__init__(config, local_rank=local_rank, world_size=world_size)
        self.cameras_drawn = []
        self.first_update = True
        self.num_imgs_to_start = config.num_imgs_to_start
        self.init_step = init_step

    def setup(self, test_mode: Literal["test", "val", "inference"] = "val"):
        """

        """
        self.pipeline = self.config.pipeline.setup(
            device=self.device,
            test_mode=test_mode,
            world_size=self.world_size,
            local_rank=self.local_rank,
            grad_scaler=self.grad_scaler,
        )  
        self.optimizers = self.setup_optimizers()

        self.dataset = self.pipeline.datamanager.train_dataset  # pyright: ignore
        
        viewer_log_path = self.base_dir / self.config.viewer.relative_log_filename
        self.viewer_state, banner_messages = None, None

        if self.config.is_viewer_enabled() and self.local_rank == 0:
            datapath = self.config.data
            if datapath is None:
                datapath = self.base_dir
                # datapath = self.config.output_dir
            self.viewer_state = ViewerState(
                self.config.viewer,
                log_filename=viewer_log_path,
                datapath=datapath,
                pipeline=self.pipeline,
                trainer=self,
                train_lock=self.train_lock,
            )
            banner_messages = [f"Viewer at: {self.viewer_state.viewer_url}"]

        self._check_viewer_warnings()

        self._load_checkpoint()

        self.callbacks = self.pipeline.get_training_callbacks(
            TrainingCallbackAttributes(
                optimizers=self.optimizers,
                grad_scaler=self.grad_scaler,
                pipeline=self.pipeline,
            )
        )

        # set up writers/profilers if enabled
        writer_log_path = self.base_dir / self.config.logging.relative_log_dir
        writer.setup_event_writer(
            self.config.is_wandb_enabled(),
            self.config.is_tensorboard_enabled(),
            log_dir=writer_log_path,
            experiment_name=self.config.experiment_name,
            project_name=self.config.project_name,
        )
        writer.setup_local_writer(
            self.config.logging, max_iter=self.config.max_num_iterations, banner_messages=banner_messages
        )
        writer.put_config(name="config", config_dict=dataclasses.asdict(self.config), step=0)
        profiler.setup_profiler(self.config.logging, writer_log_path)

    def update_data(self, images, poses):
        """

        """
        assert self.pipeline != None
        assert self.pipeline.datamanager.train_dataset is not None, "Missing DatsetInputs"

        self.pipeline.datamanager.update_train(img_list=images, transform_list=poses)

        # reallocate tensor version requires this, but causes issues so set use_preallocation = True
        if self.pipeline.datamanager.config.use_preallocation == False:
            del self.pipeline._model.field
            self.pipeline._model.field = NerfactoField(
                aabb=self.pipeline._model.scene_box.aabb,
                num_images=self.dataset.num_images,
                log2_hashmap_size=self.pipeline._model.config.log2_hashmap_size,
                max_res=self.pipeline._model.config.max_res,
                spatial_distortion=None,
            )
            self.pipeline._model.field.to(self.device)
            self.optimizers = self.setup_optimizers()
        
        # viewer_log_path = self.base_dir / self.config.viewer.relative_log_filename
        # self.viewer_state, banner_messages = None, None

        # if self.config.is_viewer_enabled() and self.local_rank == 0:
        #     datapath = self.config.data
        #     if datapath is None:
        #         datapath = self.base_dir
        #         # datapath = self.config.output_dir
        #     self.viewer_state = ViewerState(
        #         self.config.viewer,
        #         log_filename=viewer_log_path,
        #         datapath=datapath,
        #         pipeline=self.pipeline,
        #         trainer=self,
        #         train_lock=self.train_lock,
        #     )
        #     banner_messages = [f"Viewer at: {self.viewer_state.viewer_url}"]

    def train(self):
        """Train the model."""

        assert self.pipeline != None
        assert self.pipeline.datamanager.train_dataset is not None, "Missing DatsetInputs"
        assert self.pipeline.datamanager.train_image_dataloader.init_status(self.num_imgs_to_start) == True
        
        self.dataset = self.pipeline.datamanager.train_dataset  # pyright: ignore

        self._init_viewer_state()
        num_iterations = self.config.max_num_iterations
        # print('max training iteration:', num_iterations)
        step = 0
        for step in range(self._start_step, self._start_step + num_iterations):
            torch.cuda.empty_cache()
            while self.training_state == "paused":
                time.sleep(0.01)
            with self.train_lock:
                with TimeWriter(writer, EventName.ITER_TRAIN_TIME, step=step) as train_t:
                    self.pipeline.train()

                    # training callbacks before the training iteration
                    for callback in self.callbacks:
                        callback.run_callback_at_location(
                            step, location=TrainingCallbackLocation.BEFORE_TRAIN_ITERATION
                        )
                    # time the forward pass
                    loss, loss_dict, metrics_dict = self.train_iteration(step)

                    # training callbacks after the training iteration
                    for callback in self.callbacks:
                        callback.run_callback_at_location(
                            step, location=TrainingCallbackLocation.AFTER_TRAIN_ITERATION
                        )

            if step > 1:
                    writer.put_time(
                        name=EventName.TRAIN_RAYS_PER_SEC,
                        duration=self.world_size
                        * self.pipeline.datamanager.get_train_rays_per_batch()
                        / train_t.duration,
                        step=step,
                        avg_over_steps=True,
                    )

            self._update_viewer_state(step)

            # Do not perform evaluation if there are no validation images
            if self.pipeline.datamanager.eval_dataset:
                self.eval_iteration(step)

            # if step_check(step, self.config.steps_per_save, run_at_zero=True):
            #     self.save_checkpoint(step)
            
            if step_check(step, self.config.logging.steps_per_log, run_at_zero=True):
                writer.put_scalar(name="Train Loss", scalar=loss, step=step)
                writer.put_dict(name="Train Loss Dict", scalar_dict=loss_dict, step=step)
                writer.put_dict(name="Train Metrics Dict", scalar_dict=metrics_dict, step=step)
                # The actual memory allocated by Pytorch. This is likely less than the amount
                # shown in nvidia-smi since some unused memory can be held by the caching
                # allocator and some context needs to be created on GPU. See Memory management
                # (https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management)
                # for more details about GPU memory management.
                writer.put_scalar(
                    name="GPU Memory (MB)", scalar=torch.cuda.max_memory_allocated() / (1024**2), step=step
                )
                # writer.put_scalar(name=f"learning_rate/{param_group_name}", scalar=lr, step=step)
            
            if step == self.init_step:
                # CONSOLE.print(f"Saving init checkpoint to {self.config_home} at step {step}")
                save_init_checkpoint(self.config_home, self)
            
            writer.write_out_storage()
        # save checkpoint at the end of training
        self.save_checkpoint(step)

        # write out any remaining events (e.g., total train time)
        writer.write_out_storage()

        self._start_step += num_iterations

        table = Table(
            title=None,
            show_header=False,
            box=box.MINIMAL,
            title_style=style.Style(bold=True),
        )
        table.add_row("Config File", str(self.config.get_base_dir() / "config.yml"))
        table.add_row("Checkpoint Directory", str(self.checkpoint_dir))
        # CONSOLE.print(Panel(table, title="[bold][green]:tada: Training Finished :tada:[/bold]", expand=False))

        # after train end callbacks
        for callback in self.callbacks:
            callback.run_callback_at_location(step=step, location=TrainingCallbackLocation.AFTER_TRAIN)

        if not self.config.viewer.quit_on_train_completion:
            self._train_complete_viewer()


    # @check_viewer_enabled
    # def _update_viewer_state(self, step: int):
    #     """
    #     Updates the viewer state by rendering out scene with current pipeline

    #     Args:
    #         step: current train step
    #     """

    #     super()._update_viewer_state(step)
    #     #
    #     # # Clear any old cameras!
    #     if self.config.draw_training_images:

    #         camera_path = self.viewer_state.datapath / "camera_paths" 
    #         if os.path.exists(camera_path) == False:
    #             os.mkdir(camera_path)

    #         if self.first_update:
    #             # self.viewer_state.vis["sceneState/cameras"].delete()
    #             cameras = glob.glob(str(self.viewer_state.datapath) + "/camera_paths/*")
    #             for cam in cameras:
    #                 os.remove(cam)
    #             self.first_update = False

    #         # Draw any new training images
    #         image_indices = self.dataset.updated_indices
    #         for idx in image_indices:
    #             if not idx in self.cameras_drawn:
    #                 # Do a copy here just to make sure we aren't
    #                 # changing the training data downstream.
    #                 # TODO: Verify if we need to do this
    #                 image = self.dataset[idx]["image"]
    #                 bgr = image[..., [2, 1, 0]]
    #                 camera_json = self.dataset.cameras.to_json(
    #                     camera_idx=idx, image=bgr, max_size=10
    #                 )
    #                 file_path = camera_path / f"{idx:06d}.json"  
    #                 with open(str(file_path), "w") as outfile:
    #                     json.dump(camera_json, outfile)
    #                 self.cameras_drawn.append(idx)

    def train_nvf_iteration(self, step: int):
        """
        Train the model for one iteration.

        Args:
            step: current train step
        """
        pipeline = self.pipeline
        model = pipeline.model.field.visibility_head
        cameras = self.cameras
        batch_size = self.config.nvf_batch_size
        loss_fn = self.nvf_loss_fn
        train_batch_repeat = self.config.nvf_train_batch_repeat
        optimizer = self.vis_optimizer
        
        cpu_or_cuda_str: str = self.device.split(":")[0]

        torch.cuda.empty_cache()
        with torch.no_grad():
            train_points, train_visibility = gen_data(cameras, pipeline.model, batch_size)
            train_weight = get_balance_weight(train_visibility)
            loss_fn.weight = train_weight

            train_points.requires_grad = False
            density_embedding = get_density_embedding(train_points, pipeline.model).detach()

        train_points.requires_grad = True
        density_embedding.requires_grad = True
        for _ in range(train_batch_repeat):
            optimizer.zero_grad()

            train_pred = pipeline.model.field.get_visibility(train_points, density_embedding, activation=False)
            
            train_loss = loss_fn(train_pred, train_visibility)

            if self.nvf_use_var:
                with torch.autocast(device_type=cpu_or_cuda_str, enabled=self.mixed_precision):
                    _, loss_dict, metrics_dict = self.pipeline.get_train_loss_dict(step=step)
                    # loss = functools.reduce(torch.add, loss_dict.values())
                    loss = loss_dict['gmm_loss']
                train_loss += loss*self.nvf_use_var_factor

            train_loss.backward()
            optimizer.step()
        
        return train_loss.detach()
    
    def train_var_iteration(self, step: int):
        train_batch_repeat = 3#self.config.nvf_train_batch_repeat
        optimizer = self.var_optimizer
        
        cpu_or_cuda_str: str = self.device.split(":")[0]
        for _ in range(train_batch_repeat):
            optimizer.zero_grad()
            with torch.autocast(device_type=cpu_or_cuda_str, enabled=self.mixed_precision):
                _, loss_dict, metrics_dict = self.pipeline.get_train_loss_dict(step=step)
                # loss = functools.reduce(torch.add, loss_dict.values())
                train_loss = loss_dict['gmm_loss']
            train_loss.backward()
            optimizer.step()

        return train_loss.detach()

    
    def train_nvf(self):
        assert self.pipeline.model.field.use_visibility
        
        # batch_size = 65536
        # num_eval = batch_size
        # epochs = 300
        # train_batch_repeat = 3
        learning_rate = 1e-3

        batch_size = self.config.nvf_batch_size
        num_eval = self.config.nvf_batch_size
        epochs = self.config.nvf_num_iterations
        train_batch_repeat = self.config.nvf_train_batch_repeat


        self.vis_optimizer = torch.optim.Adam(self.pipeline.model.field.visibility_head.parameters(), lr=learning_rate)
        self.var_optimizer = torch.optim.Adam(self.pipeline.model.field.mlp_rgb_variance.parameters(), lr=1e-3)
        self.cameras = self.pipeline.datamanager.get_cameras()
        self.nvf_loss_fn = nn.BCEWithLogitsLoss()

        num_batch_eval = num_eval // batch_size

        eval_points, eval_visibility = gen_data(self.cameras, self.pipeline.model, num_eval)
        eval_weight = get_balance_weight(eval_visibility)
        
        self.pipeline.model.train_nvf = True
        for i in range(epochs+1):
            train_loss = self.train_var_iteration(i)
            # if i%100==0:
            #     print('var', i, train_loss.item())   
        # self.pipeline.model.train_nvf = False   

        for i in range(epochs+1):
            train_loss = self.train_nvf_iteration(i)

            if i%100==0:
                eval_loss=0.
                for _ in range(num_batch_eval):
                    density_embedding = get_density_embedding(eval_points, self.pipeline.model).detach()
                    with torch.no_grad():
                        eval_pred = self.pipeline.model.field.get_visibility(eval_points, density_embedding, activation=False)
                    self.nvf_loss_fn.weight = eval_weight
                    eval_loss += self.nvf_loss_fn(eval_pred, eval_visibility)
                eval_loss /= num_batch_eval

                # print('nvf', i, train_loss.item(), eval_loss.item())
        self.pipeline.model.train_nvf = False
        

    def reset_writer(self, relative_log_dir, use_tensorboard=None):
        if use_tensorboard is None:
            use_tensorboard = self.config.is_tensorboard_enabled()
        else:
            self.config.vis = "viewer+tensorboard" if use_tensorboard else 'viewer'
        writer.EVENT_WRITERS = []
        writer.EVENT_STORAGE = []
        writer.GLOBAL_BUFFER = {}
        writer_log_path = Path(f'{self.base_log_dir}/{relative_log_dir}')
        writer.setup_event_writer(
            self.config.is_wandb_enabled(),
            self.config.is_tensorboard_enabled(),
            log_dir=writer_log_path,
            experiment_name=self.config.experiment_name,
            project_name=self.config.project_name,
        )
        writer.setup_local_writer(
            self.config.logging, max_iter=self.config.max_num_iterations, banner_messages=None
        )
        writer.put_config(name="config", config_dict=dataclasses.asdict(self.config), step=0)

        

