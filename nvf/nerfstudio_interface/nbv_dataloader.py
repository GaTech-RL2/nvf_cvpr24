"""
Defines the TensorDataloader object that populates an image tensor and Cameras object 
based on incrementally added images/poses.
"""
import time
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
)

import numpy as np
import scipy.spatial.transform as transform
from rich.console import Console
import torch
from torch.utils.data.dataloader import DataLoader

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.process_data.colmap_utils import qvec2rotmat
import nerfstudio.utils.poses as pose_utils

from nvf.nerfstudio_interface.nbv_dataset import NBVDataset

CONSOLE = Console(width=120)

# Suppress a warning from torch.tensorbuffer about copying that
# does not apply in this case.
warnings.filterwarnings("ignore", "The given buffer")

class NBVDataloader(DataLoader):
    """
    Creates batches of the dataset return type. In this case of nerfstudio this means
    that we are returning batches of full images, which then are sampled using a
    PixelSampler. For this class the image batches are progressively growing as
    more images are added, and stored in a pytorch tensor.

    Args:
        dataset: Dataset to sample from.
        device: Device to perform computation.
    """

    dataset: NBVDataset

    def __init__(
        self,
        dataset: NBVDataset,
        device: Union[torch.device, str] = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs,
    ):
        # This is mostly a parameter placeholder, and manages the cameras
        self.dataset = dataset

        # Image meta data
        self.device = device
        self.num_images = len(self.dataset)
        self.H = self.dataset.image_height
        self.W = self.dataset.image_width
        self.n_channels = 4

        # Tracking data updates

        self.current_idx = 0
        self.updated = True

        # Keep it in the format so that it makes it look more like a
        # regular data loader.
        self.data_dict = {
            "image": self.dataset.image_tensor,
            "image_idx": self.dataset.image_indices,
        }

        super().__init__(dataset=dataset, **kwargs)

    def init_status(self, num_to_start):
        """
        Check if any image-pose pairs have been successfully added, 
        and return True if so.
        """
        return self.current_idx >= (num_to_start - 1)

    def add_data(self, image_list, pose_list, use_preallocation):
        assert len(image_list) == len(pose_list)
        
        if use_preallocation:
            if (self.current_idx + len(image_list) < self.num_images):
                for i in range(len(image_list)):
                    img = torch.from_numpy(image_list[i]).to(dtype=torch.float32).to(self.device)
                    self.dataset.image_tensor[self.current_idx] = img
                
                    pose = pose_list[i][:3,:].to(dtype=torch.float32).to(self.device)
                    self.dataset.cameras.camera_to_worlds[self.current_idx] = pose

                    self.dataset.updated_indices.append(self.current_idx)
                    self.current_idx += 1
                self.updated = True
        
        else: 
            # reallocate tensor version, but causes issues so set use_preallocation = True
            updated_image_tensor = torch.ones(
                                self.current_idx + len(image_list), 
                                self.dataset.image_height, self.dataset.image_width, 
                                4, 
                                dtype=torch.float32
                            ).to(self.device)
            updated_image_tensor[:self.current_idx] = self.dataset.image_tensor[:self.current_idx]
            del self.dataset.image_tensor
            self.dataset.image_tensor = updated_image_tensor

            updated_image_indices = torch.arange(self.current_idx + len(image_list), dtype=torch.long).to(self.device)
            del self.dataset.image_indices
            self.dataset.image_indices = updated_image_indices 

            self.data_dict = {
                "image": self.dataset.image_tensor,
                "image_idx": self.dataset.image_indices,
            }

            new_num = (self.current_idx + len(image_list))
            updated_c2w = torch.stack( new_num * [torch.eye(4, dtype=torch.float32).to(self.device)])[
                :, :-1, :
            ]
            updated_c2w[:self.current_idx] = self.dataset.cameras.camera_to_worlds[:self.current_idx]
            # del self.dataset.cameras.camera_to_worlds
            # self.dataset.cameras.camera_to_worlds = updated_c2w

            new_cam = Cameras(
                fx=self.dataset.cameras.fx[0].item(),
                fy=self.dataset.cameras.fy[0].item(),
                cx=self.dataset.cameras.cx[0].item(),
                cy=self.dataset.cameras.cy[0].item(),
                distortion_params=self.dataset.cameras.distortion_params[0].cpu(),
                height=self.dataset.cameras.height[0].item(),
                width=self.dataset.cameras.width[0].item(),
                camera_to_worlds = updated_c2w.cpu(),
                camera_type = self.dataset.cameras.camera_type[0].item(),
            ).to(self.dataset.device)
            del self.dataset.cameras
            new_cam.camera_to_worlds = new_cam.camera_to_worlds.to(self.dataset.device)
            new_cam.distortion_params = new_cam.distortion_params.to(self.dataset.device)
            self.dataset.cameras = new_cam

            for i in range(len(image_list)):
                img = torch.from_numpy(image_list[i]).to(dtype=torch.float32).to(self.device)
                self.dataset.image_tensor[self.current_idx] = img
            
                pose = torch.from_numpy(pose_list[i][:3,:]).to(dtype=torch.float32).to(self.device)
                self.dataset.cameras.camera_to_worlds[self.current_idx] = pose

                self.dataset.updated_indices.append(self.current_idx)
                self.current_idx += 1

            self.dataset.num_images = self.current_idx 
            self.updated = True

    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)
    
    def _get_updated_batch(self):
        batch = {}
        for k, v in self.data_dict.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v[: self.current_idx, ...]
        # print('batch size:', batch['image'].shape)
        return batch

    def __iter__(self):
        while True:
            if self.updated:
                self.batch = self._get_updated_batch()
                self.updated = False

            batch = self.batch
            yield batch