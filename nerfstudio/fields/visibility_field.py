from typing import Dict, Literal, Optional, Tuple, Set

import torch
from torch import Tensor, nn
from jaxtyping import Float

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.encodings import HashEncoding, NeRFEncoding, SHEncoding
from nerfstudio.field_components.field_heads import (
    FieldHeadNames,
    PredNormalsFieldHead,
    SemanticFieldHead,
    TransientDensityFieldHead,
    TransientRGBFieldHead,
    UncertaintyFieldHead,
)
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field, shift_directions_for_tcnn
from nerfstudio.field_components.base_field_component import FieldComponent

class VisibilityField(nn.Module):
    """Multilayer perceptron

    Args:
        in_dim: Input layer dimension
        num_layers: Number of network layers
        layer_width: Width of each MLP layer
        out_dim: Output layer dimension. Uses layer_width if None.
        activation: intermediate layer activation function.
        out_activation: output activation function.
    """

    def __init__(
        self,
        in_dim: int,
        num_layers: int,
        layer_width: int,
        in_dim_position: int = 3,
        num_layers_position: int = 2,        
        layer_width_position: int = 64,
        out_dim_position: int = 32,
        implementation: Literal["tcnn", "torch"] = "torch",
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.in_dim_position = in_dim_position
        assert self.in_dim > 0 and self.in_dim_position>0


        self.mlp_position = MLP(
            in_dim=in_dim_position,
            num_layers=num_layers_position,
            layer_width=layer_width_position,
            out_dim=out_dim_position,
            activation=nn.ReLU(),
            out_activation=None,
            implementation=implementation,
        )

        self.mlp_head = MLP(
            in_dim=in_dim + out_dim_position,
            num_layers=num_layers,
            layer_width=layer_width,
            out_dim=1,
            activation=nn.ReLU(),
            out_activation=None,
            implementation=implementation,
        )
        self.out_activation = nn.Sigmoid()

    def forward(self, position: Float[Tensor, "*bs in_dim_position"], embedding: Float[Tensor, "*bs in_dim"]) -> Float[Tensor, "*bs 1"]:
        position = self.mlp_position(position)
        embedding = torch.cat([embedding, position], dim=-1)
        return self.mlp_head(embedding)
