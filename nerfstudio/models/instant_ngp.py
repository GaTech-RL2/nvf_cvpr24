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

"""
Implementation of Instant NGP.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type

import matplotlib.pyplot as plt
import nerfacc
import torch
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import (TrainingCallback,
                                         TrainingCallbackAttributes,
                                         TrainingCallbackLocation)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.nerfacto_field import NerfactoField
from nvf.uncertainty.entropy_renderers import \
    WeightDistributionEntropyRenderer, RGBVarianceEntropyRenderer
from nerfstudio.model_components.losses import MSELoss

from nerfstudio.model_components.ray_samplers import VolumetricSampler, UniformSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
)
from nvf.uncertainty.entropy_renderers import (
    WeightDistributionEntropyRenderer,
    VisibilityEntropyRenderer,
    gmm_nll,
)

from nerfstudio.model_components.renderers import (AccumulationRenderer,
                                                   DepthRenderer, RGBRenderer,
                                                   RGBVarianceRenderer)
from nerfstudio.model_components.scene_colliders import NearFarCollider

from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps


@dataclass
class InstantNGPModelConfig(ModelConfig):
    """Instant NGP Model Config"""

    _target: Type = field(
        default_factory=lambda: NGPModel
    )  # We can't write `NGPModel` directly, because `NGPModel` doesn't exist yet
    """target class to instantiate"""
    enable_collider: bool = False
    """Whether to create a scene collider to filter rays."""
    collider_params: Optional[Dict[str, float]] = None
    """Instant NGP doesn't use a collider."""
    grid_resolution: int = 128
    """Resolution of the grid used for the field."""
    grid_levels: int = 4
    """Levels of the grid used for the field."""
    max_res: int = 2048
    """Maximum resolution of the hashmap for the base mlp."""
    log2_hashmap_size: int = 19
    """Size of the hashmap for the base mlp"""
    alpha_thre: float = 0.01
    """Threshold for opacity skipping."""
    cone_angle: float = 0.004
    """Should be set to 0.0 for blender scenes but 1./256 for real scenes."""
    render_step_size: Optional[float] = None
    """Minimum step size for rendering."""
    near_plane: float = 0.05
    """How far along ray to start sampling."""
    far_plane: float = 1e3
    """How far along ray to stop sampling."""
    use_appearance_embedding: bool = False
    """Whether to use an appearance embedding."""
    background_color: Literal["random", "black", "white"] = "random"
    """The color that is given to untrained areas."""
    disable_scene_contraction: bool = True
    """Whether to disable scene contraction or not."""
    use_neur_ar: bool = False
    """Use neurAR model"""
    use_active_nerf: bool = False
    """Use active nerf model"""
    use_opacity_renderer:bool = False
    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

torch.cuda.empty_cache()
class NGPModel(Model):
    """Instant NGP model

    Args:
        config: instant NGP configuration to instantiate model
    """
    config: InstantNGPModelConfig
    field: NerfactoField
    calculate_entropy: bool = False
    use_uniform_sampler: bool = False
    n_uniform_samples: int = 50
    use_nvf = False
    train_nvf = False


    def __init__(self, config: InstantNGPModelConfig, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        self.populate_entropy_modules()

    def update_embedding(self, num_images):
        self.field.update_embedding(num_images)

    def populate_entropy_modules(self):
        self.calculate_entropy = True
        if self.use_uniform_sampler == True:
            self.collider = NearFarCollider(near_plane=0.03, far_plane=8)
            self.sampler_uniform = UniformSampler(
                num_samples = self.n_uniform_samples,
            )
    def remove_entropy_modules(self):
        assert self.calculate_entropy == True

        self.calculate_entropy = False

        del self.collider 
        if hasattr(self, 'sampler_uniform'): del self.sampler_uniform

        self.collider = None
        self.sampler_uniform = None

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        self.field = NerfactoField(
            aabb=self.scene_box.aabb,
            num_images=self.num_train_data,
            log2_hashmap_size=self.config.log2_hashmap_size,
            max_res=self.config.max_res,
            spatial_distortion=scene_contraction,
            use_rgb_variance=False, # Change to enable only when NeurAR or ActiveNeRF being used
        )

        self.scene_aabb = Parameter(self.scene_box.aabb.flatten(), requires_grad=False)

        if self.config.render_step_size is None:
            # auto step size: ~1000 samples in the base level grid
            self.config.render_step_size = ((self.scene_aabb[3:] - self.scene_aabb[:3]) ** 2).sum().sqrt().item() / 1000
        # Occupancy Grid.
        self.occupancy_grid = nerfacc.OccGridEstimator(
            roi_aabb=self.scene_aabb,
            resolution=self.config.grid_resolution,
            levels=self.config.grid_levels,
        )
        # Sampler
        self.sampler = VolumetricSampler(
            occupancy_grid=self.occupancy_grid,
            density_fn=self.field.density_fn,
        )

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="expected")
        # self.renderer_entropy = WeightDistributionEntropyRenderer()
        # self.renderer_entropy = VisibilityEntropyRenderer()

        if self.config.use_active_nerf or self.config.use_neur_ar:
            self.renderer_entropy = RGBVarianceEntropyRenderer()
        else:
            self.renderer_entropy = WeightDistributionEntropyRenderer()

        if self.config.use_active_nerf or self.config.use_neur_ar:
            self.renderer_rgb_variance= RGBVarianceRenderer()
        if self.config.use_opacity_renderer:
            self.renderer_entropy = OpacitySampleEntropyRenderer()
        # self.renderer_entropy = OcclusionAwareOpacitySampleEntropyRenderer()


        # losses
        self.rgb_loss = MSELoss()
        if self.config.use_active_nerf or self.config.use_neur_ar:
            self.rgb_variance_loss = MSELoss()
        
        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        def update_occupancy_grid(step: int):
            self.occupancy_grid.update_every_n_steps(
                step=step,
                occ_eval_fn=lambda x: self.field.density_fn(x) * self.config.render_step_size,
            )

        return [
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=update_occupancy_grid,
            ),
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN],
                update_every_num_iters=1,
                func=update_occupancy_grid,
            ),
        ]

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        if self.field is None:
            raise ValueError("populate_fields() must be called before get_param_groups")
        param_groups["fields"] = list(self.field.parameters())
        return param_groups
    
    #torch.no_grad()
    #torch.cuda.empty_cache()
    def get_entropy(self, ray_bundle: RayBundle):
        assert self.field is not None
        num_rays = len(ray_bundle)

        if self.use_uniform_sampler == True:
            if ray_bundle.nears is None:
                ray_bundle.nears = torch.zeros_like(ray_bundle.metadata['directions_norm'])
            if ray_bundle.fars is None:
                ray_bundle.fars = torch.zeros_like(ray_bundle.metadata['directions_norm'])+10.
            if(ray_bundle.origins.requires_grad):
                uniform_ray_samples = self.sampler_uniform(ray_bundle)
            else:
                with torch.no_grad():
                    uniform_ray_samples = self.sampler_uniform(ray_bundle)
            field_outputs = self.field(uniform_ray_samples)

            entropy = self.renderer_entropy(
                field_outputs = field_outputs,
                ray_samples=uniform_ray_samples
            )
        else:
            # ray_bundle.origins.requires_grad_(True)
            # ray_bundle.directions.requires_grad_(True)
            ray_samples, ray_indices = self.sampler(
                ray_bundle=ray_bundle,
                near_plane=self.config.near_plane,
                far_plane=self.config.far_plane,
                render_step_size=self.config.render_step_size,
                alpha_thre=self.config.alpha_thre,
                cone_angle=self.config.cone_angle,
            ) 
            field_outputs = self.field(ray_samples) 

            entropy = self.renderer_entropy(
                field_outputs = field_outputs,
                ray_samples=ray_samples, 
                ray_indices=ray_indices, 
                num_rays=num_rays
            )
            # entropy.sum().backward()
            # print(ray_samples.frustums.origins.shape)
            # print(ray_bundle.directions.grad.sum())
            # print(ray_bundle.origins.grad.sum())
            # breakpoint()
        
        return entropy

    def get_outputs(self, ray_bundle: RayBundle):
        assert self.field is not None
        num_rays = len(ray_bundle)
        # print('instant_ngp got rays:', num_rays)
    
        with torch.no_grad():
            ray_samples, ray_indices = self.sampler(
                ray_bundle=ray_bundle,
                near_plane=self.config.near_plane,
                far_plane=self.config.far_plane,
                render_step_size=self.config.render_step_size,
                alpha_thre=self.config.alpha_thre,
                cone_angle=self.config.cone_angle,
            )
        field_outputs = self.field(ray_samples)
        # print('sampler sampled samples:', field_outputs[FieldHeadNames.DENSITY].shape)
        # accumulation
        packed_info = nerfacc.pack_info(ray_indices, num_rays)
        weights = nerfacc.render_weight_from_density(
            t_starts=ray_samples.frustums.starts[..., 0],
            t_ends=ray_samples.frustums.ends[..., 0],
            sigmas=field_outputs[FieldHeadNames.DENSITY][..., 0],
            packed_info=packed_info,
        )[0]
        weights = weights[..., None]
        rgb = self.renderer_rgb(
            rgb=field_outputs[FieldHeadNames.RGB],
            weights=weights,
            ray_indices=ray_indices,
            num_rays=num_rays,
        )
        depth = self.renderer_depth(
            weights=weights, ray_samples=ray_samples, ray_indices=ray_indices, num_rays=num_rays
        )
        accumulation = self.renderer_accumulation(weights=weights, ray_indices=ray_indices, num_rays=num_rays)
        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "num_samples_per_ray": packed_info[:, 1],
        }


        
        if self.config.use_neur_ar or self.config.use_active_nerf:
            rgb_variance = self.renderer_rgb_variance(betas=field_outputs[FieldHeadNames.RGB_VARIANCE], weights=weights,ray_indices=ray_indices, num_rays=num_rays)
            outputs["rgb_variance"] = rgb_variance

        if self.use_nvf:
            gmm, gmm_packed_info = self.renderer_entropy.get_gmm(field_outputs, ray_samples, ray_indices, num_rays)
            outputs.update({'gmm':gmm,'gmm_packed_info':gmm_packed_info})


        if self.calculate_entropy == True:
            # with torch.no_grad():
            if self.use_uniform_sampler == True:
                if self.training:
                    entropy = None
                else:
                    entropy = self.get_entropy(ray_bundle)
            else:
                entropy = self.renderer_entropy(
                    field_outputs = field_outputs,
                    ray_samples=ray_samples, 
                    ray_indices=ray_indices, 
                    num_rays=num_rays
                )
            if entropy is not None:
                outputs.update ({
                    "entropy":entropy
                })

        return outputs

    def get_metrics_dict(self, outputs, batch):
        image = batch["image"].to(self.device)
        image = self.renderer_rgb.blend_background(image)
        metrics_dict = {}
        # metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
        predicted_rgb, gt_rgb = self.renderer_rgb.blend_background_for_loss_computation(
            gt_image=image, pred_image=outputs["rgb"], pred_accumulation=outputs["accumulation"]
        )
        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)
        metrics_dict["num_samples_per_batch"] = outputs["num_samples_per_ray"].sum()
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        image = batch["image"].to(self.device)
        pred_rgb, image = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb"],
            pred_accumulation=outputs["accumulation"],
            gt_image=image,
        )

        rgb_loss = self.rgb_loss(image, pred_rgb)
        loss_dict = {"rgb_loss": rgb_loss}
        
        if self.config.use_neur_ar:
            torch.cuda.empty_cache()
            rgb_variance_gt = (image-pred_rgb)**2
            loss_dict["neur_ar_loss"] = 1e-2*self.rgb_variance_loss(outputs["rgb_variance"],rgb_variance_gt)

        if self.use_nvf and self.train_nvf:
            gmm, gmm_packed_info = outputs['gmm'], outputs['gmm_packed_info']
            gmm_loss = 0.01 * gmm_nll(image, *gmm, *gmm_packed_info)
            # if torch.isnan(gmm_loss).any():
            #     breakpoint()
            # print(gmm_loss.mean(), rgb_loss)
            
            loss_dict.update({"gmm_loss": gmm_loss.mean()})
        # if torch.isnan(rgb_loss):
        #     breakpoint()
                
        if self.config.use_active_nerf:
            rgb_variance_gt = (image-pred_rgb)**2
            self.regularization_coeff = 0.01
            # add regularization loss to ensure sparser volume density
            loss_dict["active_nerf_loss"] = torch.mean((1 / (2*(rgb_variance_gt.mean()+1e-9).unsqueeze(-1))) * rgb_variance_gt) + 0.5*torch.mean(torch.log(rgb_variance_gt.mean()+1e-9)) + self.regularization_coeff * outputs["accumulation"].mean() + 4.0
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(self.device)
        image = self.renderer_rgb.blend_background(image)
 
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )
        predicted_rgb, gt_rgb = self.renderer_rgb.blend_background_for_loss_computation(
            gt_image=image, pred_image=outputs["rgb"], pred_accumulation=outputs["accumulation"]
        )

        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)        
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]
        
        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim), "lpips": float(lpips)}  # type: ignore
        # TODO(ethan): return an image dictionary

        images_dict = {
            "img": combined_rgb,
            "accumulation": combined_acc,
            "depth": combined_depth,
        }

        return metrics_dict, images_dict
