from nvf.env.BlenderRenderer import *
from nvf.env.BlenderProcess import BlenderProcess
import nvf.env.gen_data_fn as gen_data_fn
import contextlib

class HubbleScene(BlenderGLB):
    """docstring for HubbleScene"""
    name = 'hubble'
    def __init__(self, cfg):
        super(HubbleScene, self).__init__(cfg)
        self.hdri_rotation = [0.0, 0.0, 0.0] if not hasattr(cfg, 'hdri_rotation') else cfg.hdri_rotation # wRi
        # self.hdri_path = os.path.join(cfg.root,"assets/hdri/RenderCrate-HDRI_Orbital_38_4K.hdr")
        self.hdri_path = os.path.join(cfg.root,"assets/hdri/gray_hdri.exr")

        # self.hdri_path = "assets/hdri/neon_photostudio_4k.hdr"
        [bpy.data.objects.remove(obj) for obj in bpy.data.objects if obj.type == "MESH"]
        self.add_camera()
        self.add_object('hubble', os.path.join(cfg.root,'assets/models/Hubble.glb'), scale=self.cfg.scale)
        self.config_camera()
        self.config_blender()
        self.init_rgb_world()

        self.gen_data_fn = {
            'init': lambda :gen_data_fn.hubble_init(scale=cfg.scale, n=cfg.n_init_views),
            'eval': lambda :gen_data_fn.hubble_eval(scale=cfg.scale),
            'full': lambda :gen_data_fn.hubble_full(scale=cfg.scale),
            'part': lambda :gen_data_fn.hubble_full(Tmax=20, scale=cfg.scale, pxpy=True),
        }
        # self.set_world('rgb')

    def get_mesh(self):
        mesh = trimesh.load(os.path.join(self.cfg.root,'assets/models/Hubble.glb'),force='mesh')
        return mesh 

class LegoScene(BlenderFile):
    """docstring for LegoScene"""
    name = 'lego'
    def __init__(self, cfg):
        super(LegoScene, self).__init__(cfg)
        self.load_scene(os.path.join(cfg.root, 'assets/blend_files/lego.blend'))
        self.postprocess_fn = lambda x: postprocess_mesh(x, num_faces=2, min_len=3)

        # self.config_camera()
        # self.config_blender()

        self.gen_data_fn = {
            'init': lambda :gen_data_fn.lego_init(scale=cfg.scale, n=cfg.n_init_views),
            'eval': lambda :gen_data_fn.lego_eval(scale=cfg.scale),
            'full': lambda :gen_data_fn.lego_full(scale=cfg.scale),
        }
        self.get_blender_mesh = super().get_mesh

    def get_mesh(self):
        mesh = trimesh_load_mesh(os.path.join(self.cfg.root, 'assets/blend_files/lego.ply'))
        
        return mesh
    
    def obj_filter(self, obj):
        if obj.name.startswith('Plane'):
            # breakpoint()
            return True
        return False
    
    def render(self):
        # with contextlib.redirect_stderr(open(os.devnull, "w")):
        with stdout_redirected():
            self.load_scene(os.path.join(self.cfg.root, 'assets/blend_files/lego.blend'))
            # self.load_scene(os.path.join(self.cfg.root, 'assets/blend_files/lego.blend'))
            self.set_camera_pose(self.camera_pose)
        return super(LegoScene, self).render()

class HotdogScene(BlenderFile):
    """docstring for DrumsScene"""
    name = 'hotdog'
    def __init__(self, cfg):
        super(HotdogScene, self).__init__(cfg)
        self.load_scene(os.path.join(cfg.root, 'assets/blend_files/hotdog.blend'))
        # self.postprocess_fn = lambda x: postprocess_mesh(x, num_faces=2, min_len=3)
        self.postprocess_fn = None

        self.config_camera()
        self.config_blender()

        self.gen_data_fn = {
            'init': lambda :gen_data_fn.hotdog_init(scale=cfg.scale, n=cfg.n_init_views),
            'eval': lambda :gen_data_fn.hotdog_eval(scale=cfg.scale),
            'full': lambda :gen_data_fn.hotdog_full(scale=cfg.scale),
            'full2': lambda :gen_data_fn.hotdog_full(Tmax=20, scale=cfg.scale),
            'part': lambda :gen_data_fn.hotdog_full(Tmax=20, scale=cfg.scale, pxpy=True),
        }
    
    def get_mesh(self):
        # with open('assets/blend_files/hotdog.obj', 'r') as f:
        mesh = trimesh_load_mesh(os.path.join(self.cfg.root, 'assets/blend_files/hotdog.ply'))
        
        return mesh

class RoomScene(BlenderFile):
    """docstring for RoomScene"""
    name = 'room'
    def __init__(self, cfg):
        super(RoomScene, self).__init__(cfg)
        self.cfg.n_init_views = 10
        self.set_mesh_filename(filepath=os.path.join(cfg.root, 'assets/blend_files/room.obj'))
        self.load_scene(os.path.join(cfg.root, 'assets/blend_files/room.blend'))
        self.postprocess_fn = lambda x: postprocess_mesh(x, num_faces=2, min_len=3)
        self.gen_data_fn = {
            'init': lambda :gen_data_fn.room_init(scale=cfg.scale, n=cfg.n_init_views),
            'eval': lambda :gen_data_fn.room_eval(scale=cfg.scale),
            'full': lambda :gen_data_fn.room_full(scale=cfg.scale),
            'part': lambda :gen_data_fn.room_part(scale=cfg.scale),
        }
        self.config_camera()
        self.config_blender()

        self.valid_range = np.array([[-2.,-2.,-2.], [2.,2.,2.]])*20

    def get_mesh(self):
        # Get all objects in the scene
        filename = self.get_mesh_filename()
        mesh = trimesh.load(filename,force='mesh')
        TF=np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]])
        mesh = mesh.apply_transform(TF)
        return mesh
    
def trimesh_load_mesh(path):
    mesh_or_scene = trimesh.load(path)

    # if self.cfg.scale !=1.:
    #     matrix = np.eye(4)
    #     matrix[:2, :2] *= self.cfg.scale
    #     mesh_or_scene.apply_transform(matrix)

    if isinstance(mesh_or_scene, trimesh.Scene):
        # If the scene contains multiple meshes, you have a few options:
        
        # Option 1: Get a single mesh (if you know there's only one)
        if len(mesh_or_scene.geometry) == 1:
            mesh = next(iter(mesh_or_scene.geometry.values()))

        # Option 2: Combine all meshes into one
        else:
            mesh = trimesh.util.concatenate(tuple(mesh_or_scene.geometry.values()))
    else:
        # The loaded object is already a Trimesh object
        mesh = mesh_or_scene
    return mesh
