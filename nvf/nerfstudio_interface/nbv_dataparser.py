# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data parser for instant ngp data"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple, Type

import imageio
import numpy as np
import torch
import json
import os
import shutil
from PIL import Image
import PIL 
import time

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.io import load_from_json
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class NBVDataParserConfig(DataParserConfig):
    """Instant-NGP data tensor parser config"""

    _target: Type = field(default_factory=lambda: NBVDataParser)
    """target class to instantiate"""
    data: Path = Path("data/ours/posterv2")
    """Directory or explicit json file path specifying location of data."""
    scene_scale: float = 1#0.3333
    """How much to scale the scene."""

    fov: float = 90 # in degrees
    width: int = 512
    height: int = 512
    
    """Below methods to allow yaml assignment"""
    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

@dataclass
class NBVDataParser(DataParser):
    """Instant NGP data tensor"""

    config: NBVDataParserConfig

    def __init__(self, config: NBVDataParserConfig):
        super().__init__(config=config)
        # self.data: Path = config.data
        # self.scale_factor: float = config.scale_factor
        self.aabb = config.scene_scale

    def get_dataparser_outputs(self, split="train", num_images: int = 4):
        dataparser_outputs = self._generate_dataparser_outputs(split, num_images)
        return dataparser_outputs

    def _generate_dataparser_outputs(self, split="train", num_images: int = 4):
        # if self.config.data.suffix == ".json":
        #     meta = load_from_json(self.config.data)
        #     data_dir = self.config.data.parent
        # else:
        #     meta = load_from_json( self.config.data / "transforms.json")
        #     data_dir = self.config.data

        meta = self.get_camera_params()

        camera_to_world = torch.stack(num_images * [torch.eye(4, dtype=torch.float32)])[
            :, :-1, :
        ]

        distortion_params = camera_utils.get_distortion_params(
            k1=float(0),
            k2=float(0),
            k3=float(0),
            k4=float(0),
            p1=float(0),
            p2=float(0),
        )

        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = [2., 2., 2.]
        x_scale, y_scale, z_scale = aabb_scale

        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-x_scale, -y_scale, -z_scale], [x_scale, y_scale, z_scale]], dtype=torch.float32
            )
        )

        # fl_x, fl_y = NBVDataParser.get_focal_lengths(meta)
        fl_x = meta["fl_x"]
        fl_y = meta["fl_y"]

        w, h = meta["w"], meta["h"]

        camera_type = CameraType.PERSPECTIVE
        if meta.get("is_fisheye", False):
            camera_type = CameraType.FISHEYE

        # Create a dummy Cameras object with the appropriate number
        # of placeholders for poses.
        cameras = Cameras(
            fx=float(fl_x),
            fy=float(fl_y),
            cx=float(meta.get("cx", 0.5 * w)),
            cy=float(meta.get("cy", 0.5 * h)),
            distortion_params=distortion_params,
            height=int(h),
            width=int(w),
            camera_to_worlds=camera_to_world,
            camera_type=camera_type,
        )
        # self.cam_fx = float(fl_x)
        # self.cam_fx = float(fl_y)
        # self.cam_cx = float(meta.get("cx", 0.5 * w))
        # self.cam_cy = float(meta.get("cy", 0.5 * h))
        # self.cam_distortion_params=distortion_params
        # self.cam_height = int(h)
        # self.cam_width= int(w)
        # self.camera_type


        image_filenames = []

        metadata = {
            "num_images": num_images,
            "image_height": int(h),
            "image_width": int(w),
        }

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            metadata=metadata,
            dataparser_scale=self.config.scene_scale,
        )

        return dataparser_outputs

    def get_camera_params(self) -> Dict:
        fov = self.config.fov /180.0*np.pi
        weight = self.config.width
        height = self.config.height
        f = 0.5 * weight/np.tan(fov/2)
        params = {}
        params["fl_x"] = f
        params["fl_y"] = f
        params["cx"] = weight/2.
        params["cy"] = height/2.
        params["w"] = weight
        params["h"] = height
        return params

    @classmethod
    def get_focal_lengths(cls, meta: Dict) -> Tuple[float, float]:
        """Reads or computes the focal length from transforms dict.
        Args:
            meta: metadata from transforms.json file.
        Returns:
            Focal lengths in the x and y directions. Error is raised if these cannot be calculated.
        """
        fl_x, fl_y = 0, 0

        def fov_to_focal_length(rad, res):
            return 0.5 * res / np.tan(0.5 * rad)

        if "fl_x" in meta:
            fl_x = meta["fl_x"]
        elif "x_fov" in meta:
            fl_x = fov_to_focal_length(np.deg2rad(meta["x_fov"]), meta["w"])
        elif "camera_angle_x" in meta:
            fl_x = fov_to_focal_length(meta["camera_angle_x"], meta["w"])

        if "camera_angle_y" not in meta or "y_fov" not in meta:
            fl_y = fl_x
        else:
            if "fl_y" in meta:
                fl_y = meta["fl_y"]
            elif "y_fov" in meta:
                fl_y = fov_to_focal_length(np.deg2rad(meta["y_fov"]), meta["h"])
            elif "camera_angle_y" in meta:
                fl_y = fov_to_focal_length(meta["camera_angle_y"], meta["h"])

        if fl_x == 0 or fl_y == 0:
            raise AttributeError("Focal length cannot be calculated from transforms.json (missing fields).")

        return (fl_x, fl_y)
