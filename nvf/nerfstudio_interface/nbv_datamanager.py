"""
A datamanager for the evaluation pipeline.
"""

from dataclasses import dataclass, field
from typing import Type, Dict, Tuple

from rich.console import Console

from nerfstudio.data.datamanagers import base_datamanager
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.cameras.rays import RayBundle

from nvf.nerfstudio_interface.nbv_dataset import NBVDataset
from nvf.nerfstudio_interface.nbv_dataloader import NBVDataloader
from nvf.nerfstudio_interface.nbv_dataparser import NBVDataParserConfig

CONSOLE = Console(width=120)

@dataclass
class NBVDataManagerConfig(base_datamanager.VanillaDataManagerConfig):
    """A datamanager that handles a tensor dataloader."""

    _target: Type = field(default_factory=lambda: NBVDataManager)
    dataparser: NBVDataParserConfig = NBVDataParserConfig()
    """ Must use only the custom DataParser here """
    num_training_images: int = 401
    """ Number of images to train on (for dataset tensor pre-allocation). """
    use_preallocation = True
    """ Set true for using preallocated tensor, false for reallocating tensor size for newly added images (currently doesn't work)"""

    """Below methods to allow yaml assignment"""
    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

class NBVDataManager(
    base_datamanager.VanillaDataManager
):  # pylint: disable=abstract-method
    """Essentially the VannilaDataManager from Nerfstudio except that the
    typical dataloader for training images is replaced with one that incrementally adds
    image and pose data into tensors.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: NBVDataManagerConfig
    train_dataset: NBVDataset

    def update_train(self, img_list, transform_list):
        self.train_image_dataloader.add_data(img_list, transform_list, self.config.use_preallocation)
        # if self.config.use_preallocation == False:
        #     self.train_dataset = self.train_image_dataloader.dataset
        #     self.iter_train_image_dataloader = iter(self.train_image_dataloader)
        #     self.train_pixel_sampler = self._get_pixel_sampler(
        #         self.train_dataset, self.config.train_num_rays_per_batch
        #     )
        #     self.train_camera_optimizer = self.config.camera_optimizer.setup(
        #         num_cameras=self.train_dataset.cameras.size, device=self.device
        #     )
        #     self.train_ray_generator = RayGenerator(
        #         self.train_dataset.cameras,
        #         self.train_camera_optimizer,
        #     )
        # self.iter_train_image_dataloader = iter(self.train_image_dataloader)

    def create_train_dataset(self) -> NBVDataset:
        self.train_dataparser_outputs = self.dataparser.get_dataparser_outputs(
            split="train", num_images=self.config.num_training_images
        )
        return NBVDataset(
            dataparser_outputs=self.train_dataparser_outputs, device=self.device
        )

    def setup_train(self):
        assert self.train_dataset is not None
        self.train_image_dataloader = NBVDataloader(
            self.train_dataset,
            device=self.device,
            num_workers=0,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
        )
        self.iter_train_image_dataloader = iter(self.train_image_dataloader)
        self.train_pixel_sampler = self._get_pixel_sampler(
            self.train_dataset, self.config.train_num_rays_per_batch
        )
        self.train_camera_optimizer = self.config.camera_optimizer.setup(
            num_cameras=self.train_dataset.cameras.size, device=self.device
        )
        self.train_ray_generator = RayGenerator(
            self.train_dataset.cameras,
            self.train_camera_optimizer,
        )
    
    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """
        First, checks for updates to the NBVDataloader, and then returns the next
        batch of data from the train dataloader.
        """
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        assert self.train_pixel_sampler is not None
        batch = self.train_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)
        return ray_bundle, batch

    def setup_eval(self):
        """
        Evaluation data is not implemented! This function is called by
        the parent class, but the results are never used.
        """
        pass

    def create_eval_dataset(self):
        """
        Evaluation data is not implemented! This function is called by
        the parent class, but the results are never used.
        """
        pass

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the eval dataloader."""
        CONSOLE.print("Evaluation data is not setup!")
        raise NameError(
            "Evaluation funcationality not yet implemented with ROS Streaming."
        )

    def next_eval_image(self, step: int) -> Tuple[int, RayBundle, Dict]:
        CONSOLE.print("Evaluation data is not setup!")
        raise NameError(
            "Evaluation funcationality not yet implemented with ROS Streaming."
        )

    def get_cameras(self):
        return self.train_dataset.cameras[: self.train_image_dataloader.current_idx]

        