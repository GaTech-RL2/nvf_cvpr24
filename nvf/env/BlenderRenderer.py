import os
import sys
from contextlib import contextmanager

import bpy
import mathutils
import bpycv
import gtsam
# import mathutils
import bmesh
import numpy as np
import torch
from gtsam import Point3, Pose3, Rot3
from PIL import Image

import trimesh
from tqdm import tqdm
import json

from nvf.env.utils import get_conf, pose_point_to, rgb_to_rgba, sharpness, variance_of_laplacian, load_from_json, write_to_json
from pathlib import Path

@contextmanager
def stdout_redirected(to=os.devnull):
    '''
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    '''
    fd = sys.stdout.fileno()

    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        sys.stdout.close() # + implicit flush()
        os.dup2(to.fileno(), fd) # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, 'w') # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(to=file)
        try:
            yield # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout) # restore stdout.
                                            # buffering and flags such as
                                            # CLOEXEC may be different

def outofbounds(loc,bounds):
    if loc[0] > bounds[1,0] or loc[0] < bounds[0,0] or \
                loc[1] > bounds[1,1] or loc[1] < bounds[0,1] or \
                loc[2] > bounds[1,2] or loc[2] < bounds[0,2]:
        return True
    else:
        return False

def convert_to_blender_pose(pose, return_matrix=False):
    if type(pose) is torch.Tensor:
        rot = Rot3.Quaternion(pose[3],*pose[:3])
        pose = Pose3(rot, pose[4:]).matrix()
    if type(pose) is Pose3:
        pose = pose.matrix()
    pose = mathutils.Matrix(pose)
    if return_matrix:
        return pose
    else:
        return pose.to_translation(), pose.to_euler()
    
def postprocess_mesh(mesh, num_faces=2, min_len=3):
    """Post processing mesh by removing small isolated pieces.

    Args:
        mesh (trimesh.Trimesh): input mesh to be processed
        num_faces (int, optional): min face num threshold. Defaults to 4096.
    """
    total_num_faces = len(mesh.faces)
    if num_faces is None:
        num_faces = total_num_faces // 100
    cc = trimesh.graph.connected_components(
        mesh.face_adjacency, min_len=min_len)
    # breakpoint()
    mask = np.zeros(total_num_faces, dtype=bool)
    cc = np.concatenate([
        c for c in cc if len(c) > num_faces
    ], axis=0)
    mask[cc] = True
    mesh.update_faces(mask)
    return mesh

def extract_mesh_data(obj, all_vertices, all_faces, num_existing_vertices):
    # If the object is a mesh, extract its data
    if obj.type == 'MESH':
        # Access the mesh data of the object
        mesh = obj.data

        # Get the object's world matrix
        matrix_world = obj.matrix_world

        # Create a BMesh from the mesh data
        bm = bmesh.new()
        bm.from_mesh(mesh)

        # Triangulate the mesh (convert non-triangle faces to triangles)
        bmesh.ops.triangulate(bm, faces=bm.faces[:], quad_method="BEAUTY", ngon_method="BEAUTY")

        # Iterate through the vertices of the BMesh
        for vertex in bm.verts:
            # Apply the object's world matrix to the vertex coordinates
            vertex_world = matrix_world @ vertex.co
            all_vertices.append(vertex_world)

        # Iterate through the faces of the BMesh and adjust vertex indices
        for face in bm.faces:
            # Convert local vertex indices to global ones
            face_verts = [v.index + num_existing_vertices for v in face.verts]
            all_faces.append(face_verts)

        # Update the number of existing vertices
        num_existing_vertices += len(bm.verts)

    # Recursively extract mesh data from children
    for child in obj.children:
        num_existing_vertices = extract_mesh_data(child, all_vertices, all_faces, num_existing_vertices)

    return num_existing_vertices

class BaseScene(object):
    name = ''
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.set_camera_params()

        self.gen_data_fn = {}
        self._mesh = None

        # self.cfg.scale
    def set_camera_params(self):
        self.height, self.width = self.cfg.resolution
        
        self.hfov = self.cfg.fov/180 *np.pi
        self.vfov = self.cfg.fov/180 *np.pi

        self.fx = 0.5*self.width/np.tan(self.hfov/2 )
        self.fy = 0.5*self.height/np.tan(self.vfov/2 )


    def set_camera_pose(self, wTc):
        raise NotImplementedError
    def render(self):
        raise NotImplementedError
    def get_mesh(self):
        raise NotImplementedError
    
    @property
    def intrinsic_matrix(self):
        return np.array([[self.fx, 0, self.width/2], [0, self.fy, self.height/2], [0, 0, 1]])

    @property
    def mesh(self):
        if self._mesh is None:
            print("Starts extracting mesh")
            self._mesh = self.get_mesh()
            print("Ends extracting mesh")
        return self._mesh
    def get_aabb(self):
        return np.array([self.mesh.vertices.min(axis=0), self.mesh.vertices.max(axis=0)])
    
    def render_pose(self, pose):
        self.set_camera_pose(pose)
        result = self.render()
        # img = result['mask']*result['image']
        img = rgb_to_rgba(result['image']*result['mask'], result['mask'])
        return img
    
    def render_poses(self, poses):
        img_list = []
        for i,pose in enumerate(tqdm(poses)):
        
            self.set_camera_pose(pose)
            result = self.render()
            img = rgb_to_rgba(result['image']*result['mask'], result['mask'])
            img_list.append(img)
        return img_list
    
    def get_camera_params(self):
        params = {"camera_angle_x": self.hfov,
            "camera_angle_y": self.hfov,
            "fl_x": self.fx,
            "fl_y": self.fy,
            "k1": 0.,
            "k2": 0.,
            "p1": 0.,
            "p2": 0.,
            "cx": self.width/2.,
            "cy": self.width/2.,
            "w": self.width,
            "h": self.width}
        return params
    
    def load_data(self, file, idx_start=None, idx_end=None, img_path=None, return_quat=True):
        """copies images from hubble dataset used for testing add_images()"""
        # path='cfg/initial_transforms.json'
        # hubble_dataset_path = "data/nerfstudio/hubble_mask/img"
        # hubble_dataset_path = "/home/jdill/nerfstudio/data/nerfstudio/hubble_mask/img"
        # files = Path(hubble_dataset_path).glob('*')

        transforms_dict = load_from_json(Path(file))
        params = self.get_camera_params()
        # assert params == {k:transforms_dict[k] for k in params.keys()}
        for k in params.keys():
            assert np.allclose(params[k], transforms_dict[k]), f'{k} is not equal: {params}, {transforms_dict}'
        # transforms_dict = load_from_json(Path("/home/jdill/nerfstudio/data/nerfstudio/hubble_mask/transforms/transforms.json"))
        # transforms_dict = load_from_json(Path("data/nerfstudio/hubble_mask/transforms.json"))

        np_images = []
        transforms = []
        i = 0
        if idx_start is None:
            idx_start=0
        if idx_end is None:
            idx_end=len(transforms_dict["frames"])

        for i in range(idx_start, idx_end):
            if img_path:
                image = Image.open(os.path.join(img_path,f"{i:04}.png"))
            else:
                image = Image.open(os.path.join(os.path.dirname(file),transforms_dict["frames"][i]['file_path']))
            np_image = np.array(image)
            # np_image = rgb_to_rgba(np_image)
            np_images.append(np_image)

            transform = np.asarray(transforms_dict["frames"][i]['transform_matrix'])
            # import pdb; pdb.set_trace()
            if return_quat:
                pose = Pose3(transform)  
                q = pose.rotation().toQuaternion() # w, x, y, z
                t = pose.translation()
                transforms.append(torch.FloatTensor([q.x(), q.y(), q.z(), q.w(), t[0], t[1], t[2]]))
            else:
                transforms.append(torch.FloatTensor(transform))

        return np_images, transforms
    
    def save_data(self, file, poses, images):
        assert file.endswith('.json')
        assert len(poses[0].shape) > 1 # pose in matrix form
        output_path = os.path.dirname(file)
        output_path_img = os.path.join(output_path, "img")
        os.makedirs(output_path_img, exist_ok=True)

        frames = []
        for i,pose in enumerate(poses):
            img = images[i]
            img_path = os.path.join('img', f'{i:04d}.png')

            dd = {"file_path": img_path,
                "sharpness": sharpness(img),
                "transform_matrix": pose.tolist()
                }
            frames.append(dd)
            
            im = Image.fromarray(img)
            im.save(os.path.join(output_path, img_path))

        data = self.get_camera_params()
        data.update({
        "aabb_scale": 1,
        "frames": frames
        })
        write_to_json(Path(file), data)
    

class Blender(BaseScene):
    def __init__(self, cfg=None):
        super(Blender, self).__init__(cfg)
        
        self.scene = bpy.context.scene

        

        self.postprocess_fn = None

        self.valid_range = np.array([[-2.,-2.,-2.], [2.,2.,2.]])
        # self.worlds = {}
        # self.current_world = None
        self.camera_scale = 1.

    
    def config_camera(self, camera=None):
        if camera is None:
            camera = self.camera
        camera.data.type = 'PERSP'
        camera.data.sensor_fit = 'HORIZONTAL'
        camera.data.sensor_width = 36.0
        camera.data.sensor_height = 24.0

        # height,width = self.cfg.resolution
        # fx = 0.5*self.width/np.tan(self.cfg.hfov/2)
        # fy = 0.5*self.width/np.tan(self.cfg.hfov/2)
        camera.data.lens = self.fx / self.width * camera.data.sensor_width

        # pixel_aspect = self.fy / self.fx
        # scene = bpy.data.scenes["Scene"]
        # scene.render.pixel_aspect_x = 1.0
        # scene.render.pixel_aspect_y = pixel_aspect

        # camera.data.dof_object = focus_target
        # camera.data.cycles.aperture_type = 'RADIUS'
        # camera.data.cycles.aperture_size = 0.100
        # camera.data.cycles.aperture_blades = 6
        camera.data.dof.use_dof = False
        self.scene.camera = camera

    def config_blender(self):
        # set up rendering
        render_settings = self.scene.render
        render_settings.resolution_x = self.cfg.resolution[1]
        render_settings.resolution_y = self.cfg.resolution[0]
        render_settings.resolution_percentage = self.cfg.resolution_percentage
        render_settings.use_file_extension = True
        render_settings.image_settings.file_format = 'PNG'

        self.set_engine(cycles=True)

    def set_engine(self, cycles=True):
        if cycles and self.cfg.cycles:
            bpy.context.scene.render.engine = 'CYCLES'
            self.scene.cycles.samples = self.cfg.cycles_samples
            if self.cfg.gpu:
                bpy.context.preferences.addons[
                    "cycles"
                ].preferences.compute_device_type = "CUDA"
                bpy.context.scene.cycles.device = "GPU"
            else:
                bpy.context.scene.cycles.device = "CPU"


            bpy.context.preferences.addons["cycles"].preferences.get_devices()
            print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)
            self.current_engine = 'cycles'
        else:
            bpy.context.scene.render.engine = 'BLENDER_EEVEE'
            self.current_engine = 'eevee'

    # def set_world(self, world_name):
        
    #     if self.current_world == world_name:
    #         return
    #     self.scene.world = self.worlds[world_name]
    #     self.current_world = world_name

    def add_camera(self, camera_matrix=np.eye(4)):
        rot = mathutils.Matrix(camera_matrix).to_euler()
        translation = mathutils.Matrix(camera_matrix).to_translation()
        bpy.ops.object.camera_add(location=translation, rotation=rot)
        camera = bpy.context.object
        self.camera = camera
        return camera
    
    def set_camera_pose(self, wTc, camera=None):
        self.camera_pose = wTc
        if hasattr(self, 'camera'):
            if camera is None:
                camera = self.camera
            loc, rot = convert_to_blender_pose(wTc)
            camera.location = loc * self.camera_scale
            camera.rotation_euler = rot

    def render(self):
        with stdout_redirected():
            result = bpycv.render_data()
        # bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
        rgb = np.array(result['image'])
        depth = np.array(result['depth'])
        inst = np.array(result['inst'])
        mask = np.where(inst>0, 1, 0).astype(bool)[...,None]
        return {'image': rgb, 'depth': depth, 'mask': mask}
    
    def set_mesh_filename(self, filepath):
        self.mesh_filepath=filepath

    def get_mesh_filename(self):
        if self.mesh_filepath is None:
            self.mesh_filepath = "data/assets/blend_files/room.obj"
        return self.mesh_filepath
    
    def get_mesh(self):
        if(self.cfg.scene=="RoomScene"):
            # Get all objects in the scene
            filename = self.get_mesh_filename()
            mesh = trimesh.load(filename,force='mesh')
            TF=np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]])
            mesh = mesh.apply_transform(TF)
            bounding_box = trimesh.primitives.Box(bounds=self.valid_range)
            clipped_mesh = mesh.intersection(bounding_box)
            return mesh
        
        # Get all objects in the scene
        all_objects = bpy.context.scene.objects

        # Initialize lists to store all vertices and faces
        all_vertices = []
        all_faces = []
        # Initialize a variable to keep track of the number of existing vertices
        num_existing_vertices = 0

        # Iterate through all objects
        for obj in all_objects:
            if not obj.parent:
                num_existing_vertices = extract_mesh_data(obj, all_vertices, all_faces, num_existing_vertices)
        # breakpoint()
        all_vertices = np.array(all_vertices) / self.camera_scale
        all_faces = np.array(all_faces, dtype=np.int64)
        # breakpoint()
        mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces)

        if self.postprocess_fn:
            mesh = self.postprocess_fn(mesh)

        bounding_box = trimesh.primitives.Box(bounds=self.valid_range)

        clipped_mesh = mesh.intersection(bounding_box)

        return clipped_mesh
    
class BlenderGLB(Blender):
    """docstring for Blender"""
    def __init__(self, cfg=None):
        super(BlenderGLB, self).__init__(cfg)
        # self.cfg = cfg
        # self.scene = bpy.context.scene

        self.objects = {}

        self.hdri_rotation = [0.0, 0.0, 0.0] # wRi
        self.hdri_path = None
    
    def init_rgb_world(self, name='rgb'):
        world = bpy.data.worlds.new(name=name)
        world.use_nodes = True
        # self.worlds[name] = world
        node_tree = world.node_tree
        # node_tree = bpy.data.node_groups.new(type="ShaderNodeTree", name="RGBNodeTree")
        # background_node = node_tree.nodes.new(type='ShaderNodeBackground')

        environment_texture_node = node_tree.nodes.new(type="ShaderNodeTexEnvironment")
        if self.hdri_path:
            environment_texture_node.image = bpy.data.images.load(self.hdri_path)
        self.lighting_texture_node = environment_texture_node

        mapping_node = node_tree.nodes.new(type="ShaderNodeMapping")
        self.lighting_mapping_node = mapping_node
        mapping_node.inputs["Rotation"].default_value = tuple(self.hdri_rotation)

        tex_coord_node = node_tree.nodes.new(type="ShaderNodeTexCoord")

        node_tree.links.new(tex_coord_node.outputs["Generated"], mapping_node.inputs["Vector"])
        node_tree.links.new(mapping_node.outputs["Vector"], environment_texture_node.inputs["Vector"])
        node_tree.links.new(environment_texture_node.outputs["Color"], node_tree.nodes["Background"].inputs["Color"])
        self.rgb_node_tree= node_tree
    def init_albedo_world(self, name='albedo'):
        pass
    def init_mask_world(self, name='mask'):
        pass

    def init_depth_world(self, name='depth'):
        pass
    
    def set_lighting(self, rotation=None, hdri_path=None):
        if rotation:
            if type(rotation) is Rot3:
                rotation = rotation.ypr()[::-1].tolist() # TODO: need double check
            self.lighting_mapping_node.inputs["Rotation"].default_value = tuple(rotation)
            self.hdri_rotation = tuple(rotation)
        if hdri_path:
            self.lighting_texture_node.image = bpy.data.images.load(hdri_path)
            self.hdri_path = hdri_path

    def add_object(self, key, glb_path, obj_matrix=np.eye(4), scale=1.):
        bpy.ops.import_scene.gltf(filepath=glb_path)
        obj = bpy.context.selected_objects[0]
        obj.scale = scale * np.ones(3)
        # print(obj.type)
        # input()
        inst_id = len(self.objects) +1000
        # inst_id = 1
        # bpycv.material_utils.set_vertex_color_material(obj)
        obj["inst_id"] = inst_id
        # print(bpy.context.active_object["inst_id"])
        # input()
        for i, ob in enumerate(bpy.data.objects): 
            if ob.parent == obj: 
                ob["inst_id"]= inst_id+i+1
                # ob["inst_id"]= inst_id
        with bpycv.activate_obj(obj):
            bpy.ops.rigidbody.object_add()
        self.objects[key] = obj
        

    def set_object_pose(self, key, wTo):
        obj = self.objects[key]
        loc, rot = convert_to_blender_pose(wTo)
        obj.location = loc
        obj.rotation_mode = "XYZ"
        obj.rotation_euler = rot
        pass

    # def choose_engine(self, mode):
    #     if mode in ['rgb'] and self.current_engine != 'cycles':
    #         self.set_engine(cycles=True)
    #     if mode in ['mask', 'depth'] and self.current_mode != 'eevee':
    #         self.set_engine(cycles=False)

    # def render_image(self, output_path=None, mode='rgb', return_image=True):
    #     if output_path is None:
    #         return_image=True
    #     # if self.current_mode!='rgb':
    #     #     self.config_rgb_shader()
    #     self.set_world(mode)
    #     self.choose_engine(mode)
    #     if not os.path.exists(os.path.dirname(output_path)):
    #         os.makedirs(os.path.dirname(output_path), exist_ok=True)
    #     with stdout_redirected():
    #         bpy.ops.render.render(write_still=True)
    #     bpy.data.images["Render Result"].save_render(filepath=output_path)
    #     if return_image:
    #         im = Image.open(output_path)
    #         return np.asarray(im)
    

    

class BlenderFile(Blender):
    """docstring for Blender"""
    def __init__(self, cfg=None):
        super(BlenderFile, self).__init__(cfg)
    
    def obj_filter(self, obj):
        return False

    def load_scene(self, scene_path, scale=1.):
        self.camera_scale = 1/scale
        bpy.ops.wm.open_mainfile(filepath=scene_path)
        self.scene = bpy.context.scene
        # breakpoint()
        self.camera = bpy.context.scene.camera
        
        

        # scale = 1.
        # obj = bpy.context.selected_objects[0]
        # obj.scale = scale * np.ones(3)
        # print(obj.type)
        # input()
        # inst_id = len(self.objects) +1000
        inst_id = 1000
        # inst_id = 1
        # bpycv.material_utils.set_vertex_color_material(obj)
        # obj["inst_id"] = inst_id
        # print(bpy.context.active_object["inst_id"])
        # input()
        # breakpoint()
        for i, ob in enumerate(bpy.data.objects): 
            # if ob.parent == obj: 
            # ob["inst_id"]= inst_id+i+1
            # breakpoint()
            # if ob.name.startswith('Plane'):
            #     breakpoint()
            # loc = np.array(ob.location)
            # bounds = self.valid_range *10
            # if outofbounds(loc, bounds):
            #     breakpoint()
            #     ob['inst_id'] = 0
            #     print(ob.name, 'is out of range')
            if self.obj_filter(ob):
                ob["inst_id"]= 0
            else:
                ob["inst_id"]= 1
            # ob["inst_id"]= inst_id
        # with bpycv.activate_obj(obj):
        #     bpy.ops.rigidbody.object_add()
        # self.objects[key] = obj

        # self.camera = bpy.data.objects.get("Camera")
        self.add_camera()
        self.config_camera()
        self.config_blender()
    
    def add_camera(self, camera_matrix=np.eye(4)):

        # camera = bpy.data.objects.get("Camera")
        # # Deselect all objects
        # bpy.ops.object.empty_add(location=[0.,0.,0.])
        # empty = bpy.context.active_object

        # bpy.context.view_layer.objects.active = camera
        # bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

        # # 4. Clear the camera's parent
        # camera.parent = None
        
        # # Optionally, delete the empty
        # bpy.data.objects.remove(empty)

        for obj in bpy.context.scene.objects:
            if obj.type == 'CAMERA':
                bpy.data.objects.remove(obj, do_unlink=True)

        # 2. Add a new camera to the scene
        return super().add_camera(camera_matrix)
    def render(self):
        return super().render()



def test_blender_file():
    cfg = get_conf()
    cfg.cycles.samples = 1000
    # cfg.cycles = False
    scene = BlenderFile(cfg)
    scene.load_scene('nvf/env/assets/blend_files/lego.blend')
    scene.config_camera()
    scene.config_blender()

    z = -0.34465557
    camera_matrix = pose_point_to(loc=[2, 0, z], target=[0,0,z])

    scene.set_camera_pose(camera_matrix)


    import matplotlib.pyplot as plt

    aabb = scene.get_aabb()
    print(aabb)
    scene.get_mesh()
    # breakpoint()

    result = scene.render()
    plt.figure()
    plt.subplot(1,2,1)
    # plt.imshow(result["image"])
    # print(np.uint16(result["inst"]).mean(), np.uint16(result["inst"]).max())
    # plt.imshow(np.uint16(result["inst"]))
    # plt.imshow(result['mask']*result['image'])
    plt.imshow(result['image'])
    plt.subplot(1,2,2)
    plt.imshow(result['mask'])

    plt.show()
    

def test_mesh():
    cfg = get_conf()
    cfg.cycles.samples = 500
    # cfg.cycles = False
    scene = BlenderFile(cfg)
    scene.load_scene('nvf/env/assets/blend_files/lego.blend')
    scene.postprocess_fn = lambda x: postprocess_mesh(x)
    scene.config_camera()
    scene.config_blender()
    z = -0.
    loc = [1., 1.5, 2.]
    tag = [0,0,0.]
    camera_matrix = pose_point_to(loc=loc, target=tag)
    # print(camera_matrix)

    scene.set_camera_pose(camera_matrix)

    
    # breakpoint()

    aabb = scene.get_aabb()
    print(aabb)
    trimesh = scene.mesh

    vertices, faces = trimesh.vertices, trimesh.faces

    result1 = scene.render()
    
    ax = plt.subplot(1,2,1)
    ax.imshow(result1['image'])

    ax = plt.subplot(1,2,2)
    ax.imshow(result1['mask'])

    with open(os.path.join(os.path.dirname(__file__), 'test.pkl'), 'wb') as f:
        data = {
            'result': result1,
            'camera': camera_matrix,
            'vertices': vertices,
            'faces': faces,
        }
        pkl.dump(data, f)

    plt.show()

    


def test_mesh2():
    cfg = get_conf()

    with open(os.path.join(os.path.dirname(__file__), 'test.pkl'), 'rb') as f:
        data = pkl.load(f)
        result1 = data['result']
        camera_matrix = data['camera']
        vertices = data['vertices']
        faces = data['faces']

    scene = BlenderGLB(cfg)
    [bpy.data.objects.remove(obj) for obj in bpy.data.objects if obj.type == "MESH"]
    scene.add_camera()
    # scene.add_object('hubble', os.path.join(cfg.root,'assets/models/Hubble.glb'), scale=scene.cfg.scale)
    scene.config_camera()
    scene.config_blender()
    scene.init_rgb_world()

    # Create a new mesh
    mesh = bpy.data.meshes.new(name="New_Mesh")

    # Set the mesh vertices and faces
    mesh.from_pydata(vertices.tolist(), [], faces.tolist())

    # Update the mesh with the new data
    mesh.update()

    # Create an object with the mesh and link it to the scene
    obj = bpy.data.objects.new("New_Object", mesh)
    bpy.context.collection.objects.link(obj)

    # Set the object as the active object in the scene
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)


    inst_id = 1 +1000
    # inst_id = 1
    # bpycv.material_utils.set_vertex_color_material(obj)
    obj["inst_id"] = inst_id
    # print(bpy.context.active_object["inst_id"])
    # input()
    for i, ob in enumerate(bpy.data.objects): 
        if ob.parent == obj: 
            ob["inst_id"]= inst_id+i+1
            # ob["inst_id"]= inst_id
    with bpycv.activate_obj(obj):
        bpy.ops.rigidbody.object_add()
    
    # camera_matrix2 = pose_point_to(loc=[2,0,0.], target=[0.,0.,0.])
    scene.set_camera_pose(camera_matrix)
    result2 = scene.render()

    ax = plt.subplot(2,2,1)
    ax.imshow(result1['image'])

    ax = plt.subplot(2,2,2)
    ax.imshow(result1['mask'])

    ax = plt.subplot(2,2,3)
    ax.imshow(result2['image'])

    ax = plt.subplot(2,2,4)
    ax.imshow(result2['mask'])

    diff = np.abs(result1['mask'].astype(int) - result2['mask'].astype(int)).sum()
    print('diff', diff/result1['mask'].sum())

    plt.pause(0.1)
    breakpoint()


if __name__ == "__main__":
    import pickle as pkl
    import matplotlib.pyplot as plt
    # test_blender_file()
    # test_mesh()
    test_mesh2()