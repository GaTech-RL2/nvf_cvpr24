from dataclasses import dataclass, field
from nvf.eval_utils import StrEnum
from typing import Literal, Optional, List, Tuple
import torch

class ModelType(StrEnum):
    ngp = 'instant-ngp'
    # nerfacto = 'nerfacto'

class AgentType(StrEnum):
    base = 'BaseAgent' # Base Agent
    opt = 'OptAgent' # Opt Agent

class SamplerType(StrEnum):
    base = 'BaseSampler' # Base sampler
    cf = 'CFSampler' # Collision free sampler

class MethodType(StrEnum):
    nvf = 'NVF' # our method
    wd = 'WeightDist' # Entropy estimation method proposed by Lee 2022 RAL https://arxiv.org/abs/2209.08409

class SceneType(StrEnum):
    hubble = 'HubbleScene'
    lego = 'LegoScene'
    room = 'RoomScene'
    hotdog = 'HotdogScene'

@dataclass
class EnvConfig:
    scene: SceneType = SceneType.hubble
    resolution: Tuple[int, int] = (512, 512)
    resolution_percentage: int = 100
    fov: float = 90
    hdri_rotation: Tuple[float, float, float] = (0., 0, 0)
    cycles: bool = True
    cycles_samples: int = 10000
    gpu = True
    scale: float = 1.
    horizon: int = 20
    n_init_views: int = 3
    gen_init: bool = False
    gen_eval: bool  = False
    save_data: bool  = True
    root: str = "data"

@dataclass
class ExpConfig:
    method: MethodType = MethodType.wd
    agent: AgentType = AgentType.base
    sampler: SamplerType = SamplerType.base
    model: ModelType = ModelType.ngp
    # scene: Literal["hubble", "lego", "room"] = "hubble"
    scene: SceneType = SceneType.hubble
    task: Literal["map"] = "map"
    env: EnvConfig = EnvConfig(scene=scene)
    
    n_repeat: int = 3 # number of reruns
    horizon: int = 20 # number of steps for evaluation
    # init_views: int = 10 # number of initial views
    
    n_sample: int = 512 # number of samples for the sampler
    n_opt: int = 3 # number of samples for the optimization in OptSampler
    opt_iter: int = 20 # number of iterations for the optimization in OptSampler
    opt_lr: float = 1e-4 # learning rate for the optimization in OptSampler
    check_density: bool = True # check density for Sampler
    density_threshold: float = 1e-4 # density threshold for Sampler
    
    train_iter: int = 5000 # number of iterations for NeRF training
    train_iter_last_factor: float = 1.0 # number of iterations for NeRF training last step
    train_use_tensorboard: bool = False
    use_uniform: bool = False
    use_huber: bool = True
    use_vis: bool = True
    use_var: bool = True
    mu: float = 0.95
    
    name: Optional[str] = None # experiment short name
    exp_name: Optional[str] = None # experiment full name (this param will get override)
    exp_folder: str = "./results" # experiment save folder (this param will get override)

    object_aabb: Optional[torch.Tensor] = None # object aabb shape 3x2, used for nerfacto_field
    target_aabb: Optional[torch.Tensor] = None # target aabb shape 3x2, used for sampler
    camera_aabb: Optional[torch.Tensor] = None # agent aabb shape 3x2

    d0: float = 1.0 # d0 \approx kD (see Eq.19 in the appendix)

    def __post_init__(self):
        self.env.scene= self.scene
        self.env.horizon = self.horizon

        if self.method == 'Random':
            self.agent = AgentType.random
        
        if self.scene.name == 'room':
            self.d0 = self.d0 * 6.0 # room scene has larger scale
