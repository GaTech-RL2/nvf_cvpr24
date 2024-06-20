from typing import Dict, Optional, Tuple, Type

from nerfstudio.field_components.field_heads import FieldHeadNames
from nvf.uncertainty.entropy_utils import Float, Int, Tensor
from nerfstudio.model_components.renderers import *
from nvf.uncertainty.entropy_utils import *
from nerfstudio.cameras.cameras import Cameras
import numpy as np

from nerfstudio.cameras.rays import Frustums
from nerfstudio.model_components.renderers import Float, Int, RaySamples, Tensor
from typing import List, Literal, Optional, Tuple, Type, Union
from jaxtyping import Float, Int

class BaseEntropyRenderer(nn.Module):
    """Base Entropy Renderer"""
    def __init__(self):
        super().__init__()
        self.iteration = 0
    
    def set_iteration(self, iteration: int):
        self.iteration = iteration

    def render_entropy_flattened(self, field_outputs: Dict[FieldHeadNames, Tensor],
                                 ray_samples: RaySamples, 
                                 ray_indices: Int[Tensor, "num_samples"], num_rays: int,
                                 ) -> Float[Tensor, "*bs 1"]:
        raise NotImplementedError

    def render_entropy_batched(self, field_outputs: Dict[FieldHeadNames, Tensor],
                                 ray_samples: RaySamples, 
                               ) -> Float[Tensor, "*bs 1"]:
        raise NotImplementedError

    
    def forward(
        self,
        field_outputs: Dict[FieldHeadNames, Tensor],
        ray_samples: RaySamples,
        ray_indices: Optional[Int[Tensor, "num_samples"]] = None,
        num_rays: Optional[int] = None,
        ) -> Float[Tensor, "*bs 1"]:
        """Calculate Entropy along the ray.

        Args:
            field_outputs: Output of the NeRF field.
            ray_samples: Ray Samples
        """

        if ray_samples.deltas == None:
            ray_samples.deltas = ray_samples.frustums.ends - ray_samples.frustums.starts

        if ray_indices is not None and num_rays is not None:
            entropy = self.render_entropy_flattened(field_outputs, ray_samples, ray_indices, num_rays)
        else:
            entropy = self.render_entropy_batched(field_outputs, ray_samples)
        
        return entropy

class WeightDistributionEntropyRenderer(BaseEntropyRenderer):
    """ 
    Entropy estimation method proposed by Lee et al. 2022 RAL https://arxiv.org/abs/2209.08409
    """
    def __init__(self) -> None:
        super().__init__()
        # self.background_color: BackgroundColor = background_color
        self.entropy_type='weight'

    def render_entropy_flattened(self, field_outputs: Dict[FieldHeadNames, Tensor], 
                                 ray_samples: RaySamples, ray_indices: Int[Tensor, "num_samples"], 
                                 num_rays: int) -> Float[Tensor, "*bs 1"]:
        '''Calculate entropy using flattened ray samples (used for VolumetricSampler in InstantNGP)'''
        packed_info = nerfacc.pack_info(ray_indices, num_rays)
        weights = nerfacc.render_weight_from_density(
            t_starts=ray_samples.frustums.starts[..., 0],
            t_ends=ray_samples.frustums.ends[..., 0],
            sigmas=field_outputs[FieldHeadNames.DENSITY][..., 0],
            packed_info=packed_info,
        )[0]
        weights = weights[..., None]


        comp_logweights = nerfacc.accumulate_along_rays(
                weights[..., 0], values=torch.log(torch.clamp(weights, min=1e-6)), ray_indices=ray_indices, n_rays=num_rays
            )
        accumulated_weight = nerfacc.accumulate_along_rays(
            weights[..., 0], values=None, ray_indices=ray_indices, n_rays=num_rays
        )

        bg_weight = 1 - accumulated_weight

        neg_entropy = comp_logweights + torch.log(torch.clamp(bg_weight, min=1e-6)) * bg_weight
        return -neg_entropy        
    
    def render_entropy_batched(self, field_outputs: Dict[FieldHeadNames, Tensor], 
                               ray_samples: RaySamples) -> Float[Tensor, "*bs 1"]:
        '''Calculate entropy using batched ray samples (used for uniform sampler or ProposalNetworkSampler in nerfacto)'''
        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])

        entropy = -torch.sum(weights * torch.nan_to_num(torch.log(weights+1e-9)) * ray_samples.deltas, dim=-2)

        return entropy


class RGBVarianceEntropyRenderer(BaseEntropyRenderer):
    """ 
    Estimate ray-based RGB variance by taking average of filed variance along the ray
    """
    def __init__(self) -> None:
        super().__init__()
        self.entropy_type='rgb_variance'

    def render_entropy_flattened(self, field_outputs: Dict[FieldHeadNames, Tensor], 
                                 ray_samples: RaySamples, ray_indices: Int[Tensor, "num_samples"], 
                                 num_rays: int) -> Float[Tensor, "*bs 1"]:
        '''Calculate entropy using flattened ray samples (used for VolumetricSampler in InstantNGP)'''
        packed_info = nerfacc.pack_info(ray_indices, num_rays)
        weights = nerfacc.render_weight_from_density(
            t_starts=ray_samples.frustums.starts[..., 0],
            t_ends=ray_samples.frustums.ends[..., 0],
            sigmas=field_outputs[FieldHeadNames.DENSITY][..., 0],
            packed_info=packed_info,
        )[0]
        weights = weights[..., None]

        entropy = torch.zeros(
            (num_rays, 1), device=weights.device, dtype=weights.dtype
        )
        entropy.index_add_(0, ray_indices, torch.clamp(weights * torch.nan_to_num(torch.mean(field_outputs[FieldHeadNames.RGB_VARIANCE],1,keepdim=True)),min=1e-9))
        return entropy
    
    def render_entropy_batched(self, field_outputs: Dict[FieldHeadNames, Tensor], 
                               ray_samples: RaySamples) -> Float[Tensor, "*bs 1"]:
        '''Calculate entropy using batched ray samples (used for uniform sampler or ProposalNetworkSampler in nerfacto)'''
        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        entropy = torch.sum(weights * torch.nan_to_num(torch.mean(field_outputs[FieldHeadNames.RGB_VARIANCE],2,keepdim=True)), dim=-2)
        return entropy