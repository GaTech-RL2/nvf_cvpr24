import os
import subprocess
import tempfile
import pickle as pkl
import numpy as np
import traceback  # Import the traceback module
import sys
import trimesh
import time

if __name__ == "__main__":
    path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(path)

from nvf.env.BlenderRenderer import BaseScene, stdout_redirected
from config import EnvConfig

class BlenderProcess(BaseScene):
    def __init__(self, cfg=None):
        super(BlenderProcess, self).__init__(cfg)
    
    def load_scene(self, scene_path):
        self.scene_path = scene_path
    
    def set_camera_pose(self, wTc):
        self.camera_pose = wTc
    
    def render(self):
        config ={'cfg': self.cfg}
        config.update({'camera_pose':self.camera_pose, 'scene_path':self.scene_path})
        # breakpoint()
        return start_process(config, max_retries=5, retry_sleep=1)
        
        # raise NotImplementedError
    
    def set_mesh_filename(self, filepath):
        self.mesh_filepath=filepath
    def get_mesh_filename(self):
        return self.mesh_filepath

    def get_mesh(self):
        mesh = trimesh.load(self.mesh_filepath,force='mesh')
        return mesh
    def config_camera(self):
        pass

    def config_blender(self):
        pass

def start_process(config, string=False, max_retries=1, retry_sleep=0.):
    retries = 0
    last_traceback = None  # Variable to store the last traceback
    last_output = None

    while retries <= max_retries:
        fd_output_path, output_path = tempfile.mkstemp()
        os.close(fd_output_path)

        input_path = None

        config.update({'output_path': output_path})

        if string:
            config_str = pkl.dumps(config, 0).decode('cp437')
            arg = config_str
        else:
            fd_input_path, input_path = tempfile.mkstemp(suffix='.pkl')
            os.close(fd_input_path)

            with open(input_path, 'wb') as temp_file:
                pkl.dump(config, temp_file)
            arg = input_path

        try:

            file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'render.py')
            completed_process = subprocess.run(["python", file_name, arg], check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            last_output = completed_process.stdout.decode('utf-8')  # Decode and store the output

            with open(output_path, 'rb') as result_file:
                result_data = pkl.load(result_file)

            return result_data

        except Exception as e:
            retries += 1
            last_traceback = traceback.format_exc()  # Capture the full traceback
            print(f"Attempt {retries}/{max_retries} failed. Retrying...\n{str(e)}")
            time.sleep(retry_sleep)

        finally:
            if output_path:
                os.remove(output_path)
            if input_path:
                os.remove(input_path)

        if retries >= max_retries:
            if last_output: print(f"Final attempt output:\n{last_output}")
            raise Exception(f"Failed after {max_retries} retries. Last error traceback:\n{last_traceback}")

def test_blender_process():
    cfg = EnvConfig()
    scene = BlenderProcess(cfg)
    scene.load_scene(os.path.join(cfg.root, 'assets/blend_files/materials.blend'))

    camera_matrix = pose_point_to(loc=[0.01, 0.0, 1.5], target=[0,0,0.])
    print(camera_matrix)
    # scene.set_camera_pose(camera_matrix)
    # print(scene.camera_pose)
    result = scene.render_pose(camera_matrix)

    im = Image.fromarray(result)
    im.save("scripts/output_temp/result.png")


if __name__ == "__main__":
    from nvf.env.utils import pose_point_to
    from PIL import Image
    pass
    # Example usage
    test_blender_process()
    # pose_matrix = np.eye(4)
    # result = start_process({'pose':pose_matrix}, string=False, max_retries=1)

    # print(result)