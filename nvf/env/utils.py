import numpy as np
# import hydra
from omegaconf import OmegaConf
import os
from PIL import Image
import glob
from nerfstudio.utils.io import load_from_json
from pathlib import Path
from gtsam import Pose3, Rot3
import torch
import random
import string
import tempfile
from scipy.spatial.transform import Rotation
from PIL import ImageDraw, ImageFont
import gc
import cv2

def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def sharpness(image):
	# image = cv2.imread(imagePath)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    fm = variance_of_laplacian(gray)
    return fm

def pose_point_to(loc, target=[0,0,0.], up=[0,0,1.]):
    x0 = np.array(target)
    p = np.array(loc)
    up = np.array(up)
    up = up/np.linalg.norm(up)
    ez = (p-x0)/ np.linalg.norm(p-x0)
    ex = np.cross(up, ez)
    if np.linalg.norm(ex)<1e-7:
        up = up+np.array([0.1, 0, 0])
        up = up/np.linalg.norm(up)
        ex = np.cross(up, ez)
    ex = ex/np.linalg.norm(ex)
    ey = np.cross(ez, ex)
    ey = ey/np.linalg.norm(ey)
    # print(np.linalg.norm(ex), np.linalg.norm(ey), np.linalg.norm(ez))
    rot = np.array([ex, ey, ez]).T
    pose = np.eye(4)
    pose[:3,:3] = rot
    pose[:3,3] = p
    return pose

def pose2tensor(pose):
    if type(pose) is not Pose3:
        pose = Pose3(pose)  
    q = pose.rotation().toQuaternion() # w, x, y, z
    t = pose.translation()
    return torch.FloatTensor([q.x(), q.y(), q.z(), q.w(), t[0], t[1], t[2]])

def tensor2pose(pose):
    pose = pose.detach().cpu().numpy()
    quat = pose[:4]
    pos = pose[-3:]
    return Pose3(Rot3.Quaternion(quat[-1], *quat[:3]), pos)


def gen_arch(pose1, pose2, center, n=10, return_matrix=True):
    if type(pose1) is not Pose3:
        pose1 = Pose3(pose1)
    if type(pose2) is not Pose3:
        pose2 = Pose3(pose2)
    if type(center) is not np.ndarray:
        center = np.array(center)
    p1 = pose1.translation()
    p2 = pose2.translation()

    a = p1 - center
    ra = np.linalg.norm(a)
    ea = a / ra

    v2 = p2 - center

    theta = np.arccos (np.dot(a, v2) / ra / np.linalg.norm(v2) )

    en = np.cross(a, v2) / ra / np.linalg.norm(v2)
    eb = np.cross(en, a) / ra

    # breakpoint()
    if np.linalg.norm(v2) > ra * np.abs(np.cos(theta)):
        rb = np.sqrt( (np.linalg.norm(v2)**2 - ra**2 * np.cos(theta)**2) / np.sin(theta)**2 ) 
    else:
        print('Ill-posed ellipse, use circle instead')
        rb = ra

    b = center + eb*rb

    thetas = np.linspace(0, theta, n).reshape(-1,1)

    points = ea.reshape(1,3)*ra*np.cos(thetas) + eb.reshape(1,3)*rb*np.sin(thetas) + center
    
    # breakpoint()
    # return points
    rot1 = pose1.rotation()
    rot2 = pose2.rotation()

    rot = [rot1.slerp(tt, rot2) for tt in np.linspace(0, 1., n)]

    pose_list = [Pose3(r, p) for r,p in zip(rot, points)]

    # breakpoint()
    if return_matrix:
        return [pose.matrix() for pose in pose_list]
    else:
        return pose_list


    

def get_conf(path=None, name='gen_nerf_data'):
    c = []
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cfg')
    
    # hydra.main(path, config_name=name)(lambda x:c.append(x))()
    # cfg = c[0]
    
    cfg = OmegaConf.load(os.path.join(path, f'{name}.yaml'))

    return cfg

def save_gif(img_list,output, duration=0.5):
    if type(img_list) is str:
        file_list = glob.glob(os.path.join(img_list, '*.png'))
        file_list.sort()
        img_list = []
        for path in file_list:
            img_list.append(np.array(Image.open(path)))
    for i,img in enumerate(img_list):
        if type(img) is np.ndarray:
            if img.max()<=1.:
                img = np.uint8(img*255)
            img = Image.fromarray(img)
        img_list[i] = img
#     img_list[0].save
    img_list[0].save(output, save_all=True, append_images=img_list[1:], duration=int(duration*1000), loop=0)
    pass

def get_transforms(idx_start=None, idx_end=None, path='cfg/initial_transforms.json', return_quat=True):
    transforms_dict = load_from_json(Path(path))
    transforms = []
    if idx_start is None:
        idx_start=0
    if idx_end is None:
        idx_end=len(transforms_dict["frames"])
    for i in range(idx_start, idx_end):
        transform = np.asarray(transforms_dict["frames"][i]['transform_matrix'])
        if return_quat:
            pose = Pose3(transform)  
            q = pose.rotation().toQuaternion() # w, x, y, z
            t = pose.translation()
            transforms.append(torch.FloatTensor([q.x(), q.y(), q.z(), q.w(), t[0], t[1], t[2]]))
        else:
            transforms.append(torch.FloatTensor(transform))
    return transforms

def is_rgb(image):
    """Check if the image is in RGB format."""
    return image.ndim == 3 and image.shape[2] == 3

def rgb_to_rgba(image, mask):
    """Convert an RGB image to RGBA."""
    if not is_rgb(image):
        raise ValueError("The input image is not in RGB format.")
    
    height, width, _ = image.shape
    
    if image.dtype == np.uint8:
        alpha_channel = np.ones((height, width, 1), dtype=np.uint8) * 255 *mask
        
    elif np.issubdtype(image.dtype, np.floating):
        alpha_channel = np.ones((height, width, 1), dtype=image.dtype)*mask
    else:
        raise ValueError("Image dtype not supported. Expected uint8 or float.")
    return np.concatenate((image, alpha_channel), axis=2).astype(image.dtype)


def get_images(idx_start=None, idx_end=None, file='data/nerfstudio/hubble_mask/transforms.json', img_path=None, return_quat=True):
    """copies images from hubble dataset used for testing add_images()"""
    # path='cfg/initial_transforms.json'
    # hubble_dataset_path = "data/nerfstudio/hubble_mask/img"
    # hubble_dataset_path = "/home/jdill/nerfstudio/data/nerfstudio/hubble_mask/img"
    # files = Path(hubble_dataset_path).glob('*')

    transforms_dict = load_from_json(Path(file))
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

# def get_images2(idx_start, idx_end):
#     """copies images from hubble dataset used for testing add_images()"""

#     hubble_dataset_path = "/home/jdill/nerfstudio/data/nerfstudio/hubble_mask/img"
#     files = Path(hubble_dataset_path).glob('*')

#     transforms_dict = load_from_json(Path("/home/jdill/nerfstudio/data/nerfstudio/hubble_mask/transforms.json"))

#     np_images = []
#     transforms = []
#     i = 0

#     for i in range(idx_start, idx_end):
#         image = Image.open(f"/home/jdill/nerfstudio/data/nerfstudio/hubble_mask/img/{i:04}.png")
#         np_image = np.array(image)
#         np_images.append(np_image)

#         transform = np.asarray(transforms_dict["frames"][i]['transform_matrix'])

#         q = Rotation.from_matrix(transform[0:3, 0:3]).as_quat()
#         t = transform[:,3]    

#         transforms.append(np.asarray([q[0], q[1], q[2], q[3], t[0], t[1], t[2]]))

#     # print(len(np_images))
#     # print(len(transforms))

#     return np_images, transforms

def save_img(img, filepath):
    path = os.path.dirname(filepath)
    if not os.path.exists(path):
        os.makedirs(path)
    if img.max()>1:
        img = Image.fromarray(np.uint8(img))
    else:
        img = Image.fromarray(np.uint8(img*255))
    img.save(filepath)

class GIFSaver(object):
    """docstring for GIFSaver"""
    def __init__(self, name=None, path=None, temp_format="png"):
        super(GIFSaver, self).__init__()
        # self.arg = arg
        # self.count=0
        if name is not None:
            self.name = name
            # self.isname = True
        else:
            # self.isname = False
            self.name = ''.join(random.choice(string.ascii_uppercase) for i in range(6))
        self.path = path
        self.temp_format = temp_format.lower()
        self.temp_path = tempfile.gettempdir()
        self.file_list = []
        self.fig_list = []
        self.count=0

    def __call__(self,count=None):
        if count is None:
            count=self.count
        fname = 'gif_tmp_'+self.name+f'_{count}.{self.temp_format}'
        fpath = os.path.join(self.temp_path, fname)
        self.file_list.append(fpath)
        self.count+=1
        return fpath
    
    def add(self, img, fn=None):
        if img.shape[-1]>3:
            img = img[...,:3] # remove alpha channel
        if img.max()>1:
            img = Image.fromarray(np.uint8(img))
        else:
            img = Image.fromarray(np.uint8(img*255))
        if fn:
            fn(img)
        self.fig_list.append(img)
        self.count+=1

    def save(self,name=None,path=None, duration=500, loop=0):
        if name :
            if os.sep in name:
                output_path = name
            elif path is not None:
                output_path = os.path.join(path, name)
            elif self.path is not None:
                output_path = os.path.join(self.path, name)
            else:
                output_path = os.path.join(os.getcwd(), name)
        else:
            if path is not None:
                output_path = os.path.join(path, self.name)
            elif self.path is not None:
                output_path = os.path.join(self.path, self.name)
            else:
                output_path = os.path.join(os.getcwd(), self.name)
        if not output_path.endswith('.gif'):
            output_path+='.gif'

        if not self.fig_list:
            images=[]
            for img_file in self.file_list:
                im = Image.open(img_file)
                images.append(im)
        else:
            images = self.fig_list
        assert len(images)>=2, 'Need at least two images in the list'
        images[0].save(os.path.join(output_path), save_all=True, append_images=images[1:], duration=duration, loop=loop)
        for img_file in self.file_list:
            os.remove(img_file)

def empty_cache():
    torch.cuda.empty_cache(); gc.collect()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def reshape_img_matrix(image_list, shape):
    assert len(shape) == 2
    idx = np.arange(len(image_list))

    new_idx = np.reshape(idx, shape)

    shape = new_idx.shape

    # n_rows = shape[0]
    # n_cols = shape[1]

    img_matrix = []

    for i in range(shape[0]):
        img_matrix.append(image_list[ new_idx[i,0]: new_idx[i,-1]+1])
    return img_matrix



def stack_img(img_mat,sep,gap_rgb=[255,255,255], shape=None):
    '''
    stack img to matrix
    img_mat:[[img1,img2],[img3,img4]]
    sep: (gap_size_height, gap_size_weight)
    '''
    if shape is not None:
        img_mat = reshape_img_matrix(img_mat, shape)

    if img_mat[0][0].max()<=1:
        gap_rgb=np.array(gap_rgb)/255.
    
    img_lines = []
    if type(img_mat) is list:
        num_row = len(img_mat)
    else:
        num_row=img_mat.shape[0]
    for j in range(num_row):
        line = img_mat[j]
        line_new = []
        for i in range(len(line)):
            if i==0:
                w_gap_h = line[i].shape[0]
                w_gap = np.ones((w_gap_h,sep[1],3))
                w_gap[...,:] = np.array(gap_rgb)
                w_gap = w_gap.astype(line[i].dtype)
                line_new.append(line[i])
            else:
                line_new.append(w_gap)
                line_new.append(line[i])
        # print([s.shape for s in line_new])
        img_line = cv2.hconcat(line_new)
        if j==0:
            h_gap_w = img_line.shape[1]
            h_gap = np.ones((sep[0],h_gap_w,3))
            h_gap[...,:] = np.array(gap_rgb)
            h_gap = h_gap.astype(img_line.dtype)
        else:
            img_lines.append(h_gap)
        img_lines.append(img_line)
    img_mat = cv2.vconcat(img_lines)
    return img_mat

def image_add_title(image, title, title_height=None, font_scale=1, font_color=(0, 0, 0), bg_color=(255, 255, 255)):
    # Create a new image with extra space for the title
    # The new image has the same width but increased height
    if image.max()<=1.:
        image = (image*255).astype(np.uint8)
    h, w = image.shape[:2]

    if title_height is None:
        title_height = int(h / 10)
    new_h = h + title_height
    new_image = np.full((new_h, w, 3), bg_color, dtype=np.uint8)

    # Copy the original image onto the new canvas
    new_image[title_height:h + title_height, :] = image

    # Choose the font
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Calculate text size and position
    text_size = cv2.getTextSize(title, font, font_scale, 2)[0]
    text_x = (w - text_size[0]) // 2
    text_y = (title_height + text_size[1]) // 2

    # Put the text onto the new image
    cv2.putText(new_image, title, (text_x, text_y), font, font_scale, font_color, 2)

    return new_image

import json
def load_from_json(filename: Path):
    """Load a dictionary from a JSON filename.

    Args:
        filename: The filename to load from.
    """
    assert filename.suffix == ".json"
    with open(filename, encoding="UTF-8") as file:
        return json.load(file)


def write_to_json(filename: Path, content: dict):
    """Write data to a JSON file.

    Args:
        filename: The filename to write to.
        content: The dictionary data to write.
    """
    assert filename.suffix == ".json"
    with open(filename, "w", encoding="UTF-8") as file:
        json.dump(content, file)


if __name__ == "__main__":
    # p1 = pose_point_to(loc=[5.3, -8, 5], target=[0,0,2.])
    p1 = pose_point_to(loc=[8, 0, 2.4], target=[-4,0,0.3])
    p2 = pose_point_to(loc=[-3.8,5.6,2.4], target=[-4,0,0.3])
    print(gen_arch(p1,p2,[-4,0,0.3]))