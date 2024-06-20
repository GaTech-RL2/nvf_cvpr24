import sys
import os
import numpy as np
path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(path)
# print(sys.path)
from config import EnvConfig

import pickle as pkl
from nvf.env.BlenderRenderer import BlenderFile

# def child_render(output_path, pose):
#     # Implement the rendering logic here and optionally write result to `output_path`
#     pass

def main():
    config_arg = sys.argv[1]

    # Check if the argument is a path to a .pkl file
    if config_arg.endswith('.pkl'):
        # Read the configuration from the .pkl file
        with open(config_arg, 'rb') as config_file:
            config = pkl.load(config_file)
    else:
        # Decode the configuration from the string argument
        config = pkl.loads(config_arg.encode('cp437'))
    
    cfg = config['cfg']
    # Extract configuration
    output_path = config['output_path']
    camera_pose = config['camera_pose']
    scene_path = config['scene_path']
    # print(sb)
    # Call the rendering function (assuming output_path is optional or handled within)
    # child_render(output_path, pose, output_path=config_arg if config_arg.endswith('.pkl') else None)

    scene = BlenderFile(cfg)
    scene.load_scene(scene_path)
    scene.config_camera()
    scene.config_blender()

    # scene.load_scene(scene_path)
    scene.set_camera_pose(camera_pose)
    result = scene.render()
    with open(output_path, 'wb') as temp_file:
        pkl.dump(result, temp_file)

if __name__ == "__main__":
    main()