from nvf.uncertainty.base_entropy_renderers import *


class VisibilityEntropyRenderer(BaseEntropyRenderer):

    beta = 0.95
    density_factor = 0.01
    sigma_un = 1.
    sigma_bg = 0.01
    rgb_un = torch.FloatTensor([0.5, 0.5, 0.5])
    rgb_bg = torch.FloatTensor([0., 0., 0.])
    d0 = 2.
    depth_decay_factor = 2
    use_huber = True
    use_visibility = True
    use_var = True
    depth_use_equiv = True
    use_depth_corr = False
    

    def __init__(self) -> None:
        super().__init__()
        self.entropy_type='visibility'

        self.depth_renderer = DepthRenderer(method="expected")

    def get_equiv_density(self, density: Float[Tensor, "*bs 1"], 
                          visibility: Float[Tensor, "*bs 1"], ray_samples: RaySamples) -> Float[Tensor, "*bs 1"]:
        # delta_density = self.deltas * density
        if not self.use_visibility:
            return density
        alpha =( visibility + self.beta * (1-visibility) ) * ( 1 - torch.exp(- (ray_samples.deltas * density)) )
        alpha += (1-self.beta) * (1 - visibility) * (1 - np.exp(- self.density_factor*self.sigma_un))

        equiv_delta_density = - torch.nan_to_num(torch.log(1-alpha))

        equiv_density = equiv_delta_density / ray_samples.deltas

        return torch.nan_to_num(equiv_density)
        # return equiv_density
    
    def get_gmm_batched(self, rgb : Float[Tensor, "*bs num_samples 3"], 
                        var_rgb : Float[Tensor, "*bs num_samples 3"], 
                        weight_eqiv: Float[Tensor, "*bs num_samples 1"],
                        visibility: Float[Tensor, "*bs num_samples 1"],
                        ) -> Tuple[Float[Tensor, "*bs num_samples+2 3"], Float[Tensor, "*bs num_samples+2 3"], Float[Tensor, "*bs num_samples+2 1"]]:
        '''
        output: means, variance, weights
        '''
        # get GMM
        weight_1 = weight_eqiv * visibility
        weight_un = weight_eqiv * (1 - visibility)
        weight_un = weight_un.sum(dim=-2, keepdim=True)

        # breakpoint()
        assert var_rgb.shape[-1] == 3

        weight_bg = torch.clip(1 - torch.sum(weight_eqiv, dim=-2, keepdim=True), min=0., max=1.)

        weight = torch.cat([weight_1, weight_bg, weight_un], dim=-2)
        
        var_un = torch.ones_like(var_rgb[..., [0],:])  * self.sigma_un **2
        var_bg = torch.ones_like(var_rgb[..., [0],:])  * self.sigma_bg **2

        # breakpoint()
        variance = torch.cat([var_rgb, var_bg, var_un], dim=-2)

        rgb_un = torch.ones_like(rgb[..., [0], :]) * self.rgb_un.to(rgb)
        rgb_bg = torch.ones_like(rgb[..., [0], :]) * self.rgb_bg.to(rgb)
        mean = torch.cat([rgb, rgb_bg, rgb_un], dim=-2)

        return mean, variance, weight
    
    def get_gmm_flatten(self, rgb : Float[Tensor, "num_samples 3"], 
                        var_rgb : Float[Tensor, "num_samples 3"], 
                        weight_eqiv: Float[Tensor, "num_samples 1"],
                        visibility: Float[Tensor, "num_samples 1"],
                        ray_indices: Int[Tensor, "num_samples"],
                        num_rays: int,
                        ) -> Tuple[Float[Tensor, "num_samples+2*num_rays 3"], Float[Tensor, "num_samples+2*num_rays 3"], Float[Tensor, "num_samples+2*num_rays 1"], Int[Tensor, "num_samples+2*num_rays"]]:
        '''
        return flatten means, variance, weights, ray_indices
        not sorted by index!
        '''
        weight_1 = weight_eqiv * visibility
        # weight_un = weight_eqiv * (1 - visibility)
        # weight_un = weight_un.sum(dim=-2, keepdim=True)
        weight_un = torch.zeros(num_rays, 1, device=weight_eqiv.device, dtype=weight_eqiv.dtype)
        weight_un = weight_un.index_add_(-2, ray_indices, weight_eqiv * (1 - visibility))

        assert var_rgb.shape[-1] == 3

        weight_sum = torch.zeros(num_rays, 1, device=weight_eqiv.device, dtype=weight_eqiv.dtype)
        weight_sum = weight_sum.index_add_(-2, ray_indices, weight_eqiv)

        weight_bg = torch.clip(1 - weight_sum, min=0., max=1.)

        weight = torch.cat([weight_1, weight_bg, weight_un], dim=-2)
        
        var_un = torch.ones(num_rays, 3).to(var_rgb) * self.sigma_un **2
        var_bg = torch.ones(num_rays, 3).to(var_rgb) * self.sigma_bg **2

        variance = torch.cat([var_rgb, var_bg, var_un], dim=-2)

        rgb_un = torch.ones(num_rays, 3).to(rgb) * self.rgb_un.to(rgb)
        rgb_bg = torch.ones(num_rays, 3).to(rgb) * self.rgb_bg.to(rgb)
        mean = torch.cat([rgb, rgb_bg, rgb_un], dim=-2)

        idx_add = torch.LongTensor(np.arange(num_rays)).to(ray_indices)
        ray_indices = torch.cat([ray_indices, idx_add, idx_add])

        return mean, variance, weight, ray_indices

    def get_gmm(self, 
            field_outputs: Dict[FieldHeadNames, Tensor],
            ray_samples: RaySamples,
            ray_indices: Optional[Int[Tensor, "num_samples"]] = None,
            num_rays: Optional[int] = None,
            )-> Float[Tensor, "*bs 1"]:
        '''
        return negative log likelihood: used for
        '''
        flatten = True if ray_indices is not None and num_rays is not None else False

        if ray_samples.deltas == None:
            ray_samples.deltas = ray_samples.frustums.ends - ray_samples.frustums.starts

        min_entropy = 0.5 * (3 * np.log(2 * np.pi * np.e) + 3* np.log (self.sigma_bg **2))


        density = field_outputs[FieldHeadNames.DENSITY]
        visibility = field_outputs[FieldHeadNames.VISIBILITY]
        
        rgb = field_outputs[FieldHeadNames.RGB]

        if self.use_var:
            rgb_var = field_outputs[FieldHeadNames.RGB_VARIANCE]
        else:
            rgb_var = torch.ones_like(rgb) * self.sigma_bg **2

        equiv_density = self.get_equiv_density(density, visibility, ray_samples)
        
        if flatten:
            if ray_samples.frustums.origins.shape[0]==1:
                # not point sampled return min entropy
                # return torch.ones(num_rays, 1).to(rgb)*min_entropy
                gmm_mean = torch.ones(num_rays, 3).to(rgb) * self.rgb_bg.view(1,3).to(rgb)
                gmm_var = torch.ones(num_rays, 3).to(rgb) * self.sigma_bg **2
                gmm_weight = torch.ones(num_rays, 1).to(rgb)
                gmm = (gmm_mean, gmm_var, gmm_weight)
                ray_indices = torch.LongTensor(np.arange(num_rays)).to(ray_indices)

            else:
                packed_info = nerfacc.pack_info(ray_indices, num_rays)
                weight_eqiv = nerfacc.render_weight_from_density(
                    t_starts=ray_samples.frustums.starts[..., 0],
                    t_ends=ray_samples.frustums.ends[..., 0],
                    sigmas=equiv_density[..., 0],
                    packed_info=packed_info,
                )[0]
                weight_eqiv = weight_eqiv[..., None]
                # if torch.any(torch.isnan(weight_eqiv)):
                #     breakpoint()
                gmm_output = self.get_gmm_flatten(rgb, rgb_var, weight_eqiv, visibility ,ray_indices, num_rays)
                gmm = gmm_output[:3]

                ray_indices = gmm_output[-1]
        else:
            weight_eqiv = ray_samples.get_weights(equiv_density)
            gmm = self.get_gmm_batched(rgb, rgb_var, weight_eqiv, visibility)
            

        return gmm, (ray_indices, num_rays)
        
    
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

        flatten = True if ray_indices is not None and num_rays is not None else False

        if ray_samples.deltas == None:
            ray_samples.deltas = ray_samples.frustums.ends - ray_samples.frustums.starts

        min_entropy = 0.5 * (3 * np.log(2 * np.pi * np.e) + 3* np.log (self.sigma_bg **2))


        density = field_outputs[FieldHeadNames.DENSITY]
        visibility = field_outputs[FieldHeadNames.VISIBILITY]
        rgb_var = field_outputs[FieldHeadNames.RGB_VARIANCE]
        rgb = field_outputs[FieldHeadNames.RGB]

        if not self.use_visibility:
            visibility = torch.ones_like(visibility)
        
        if not self.use_var:
            rgb_var = torch.ones_like(rgb) * self.sigma_bg **2

        

        equiv_density = self.get_equiv_density(density, visibility, ray_samples)
        
        if flatten:
            if ray_samples.frustums.origins.shape[0]==1:
                # not point sampled return min entropy
                return torch.ones(num_rays, 1).to(rgb)*min_entropy
            # raise NotImplementedError
            packed_info = nerfacc.pack_info(ray_indices, num_rays)
            weight_eqiv = nerfacc.render_weight_from_density(
                t_starts=ray_samples.frustums.starts[..., 0],
                t_ends=ray_samples.frustums.ends[..., 0],
                sigmas=equiv_density[..., 0],
                packed_info=packed_info,
            )[0]
            weight_eqiv = weight_eqiv[..., None]
            gmm_output = self.get_gmm_flatten(rgb, rgb_var, weight_eqiv, visibility ,ray_indices, num_rays)
            gmm = gmm_output[:3]

            ray_indices = gmm_output[-1]
        else:
            weight_eqiv = ray_samples.get_weights(equiv_density)
            gmm = self.get_gmm_batched(rgb, rgb_var, weight_eqiv, visibility)
        if torch.isnan(gmm[-1]).any():
            pass
        
        if self.use_huber:
            entropy = gmm_entropy_upper_bound_huber(*gmm, ray_indices=ray_indices, num_rays=num_rays)
        else:
            entropy = gmm_entropy_upper_bound(*gmm, ray_indices=ray_indices, num_rays=num_rays)

        if self.d0>0.:
            if self.depth_use_equiv:
                depth = self.depth_renderer(weight_eqiv, ray_samples)
            else:
                if flatten:
                    packed_info = nerfacc.pack_info(ray_indices, num_rays)
                    weight_orig = nerfacc.render_weight_from_density(
                        t_starts=ray_samples.frustums.starts[..., 0],
                        t_ends=ray_samples.frustums.ends[..., 0],
                        sigmas=density[..., 0],
                        packed_info=packed_info,
                        )[0]
                    weight_orig = weight_orig[..., None]
                else:
                    weight_orig = ray_samples.get_weights(density)
                
                depth = self.depth_renderer(weight_orig, ray_samples)
            
            # factor = torch.ones_like(depth)
            # factor[depth<self.d0] = depth[depth<self.d0] 
            if self.use_depth_corr:
                factor = 1 - torch.exp(-depth/self.d0)
            else:
                factor = torch.clip(depth/self.d0, 0., 1.)**self.depth_decay_factor

            entropy = (entropy - min_entropy ) *factor + min_entropy

        entropy = entropy.clip(min = min_entropy)

        return entropy
