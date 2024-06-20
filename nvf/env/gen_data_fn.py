from nvf.env.utils import get_conf, pose_point_to, rgb_to_rgba, get_transforms
import numpy as np
import os

def hubble_full(Tmax=400, scale=1., pxpy=False):
    trajs=  []
    for _ in range(Tmax):
        if pxpy:
            theta, phi = np.random.rand()*np.pi/2, (np.random.rand()*1/2 + 1/4.)*np.pi
        else:
            theta, phi = np.random.rand()*np.pi*2, np.random.rand()*np.pi
        point = 2*np.array([5, 5, 7.5])*np.array([np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)]) + np.array([0,0,3.])
        t_x = np.random.rand()-0.5
        t_y = np.random.rand()-0.5
        t_z = np.random.rand()*3.5-1.
        target = np.array([t_x,t_y,t_z])*2
        pose = pose_point_to(loc=point*scale, target=target*scale, up=np.random.rand(3))
        # pose[:2, 3] = pose[:2, 3]/2
        trajs.append(pose)
    return trajs

def hubble_eval(scale=1.):
    def get_pose(theta, phi):
        target = np.array([0,0,2.0])
        point = 2*np.array([5.2, 5.2, 7.5])*np.array([np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)]) + target
        
        pose = pose_point_to(loc=point*scale, target=target*scale)
        return pose
    trajs = []
    n1, n2 = 3, 6 #6, 8 # 3, 6
    altitude_list = np.linspace(0, np.pi, n1+2)[1:-1]
    longitude_list = np.linspace(0, np.pi*2, n2+1)[:-1]
    for theta in  longitude_list:
        for phi in altitude_list:
            pose = get_pose(theta,phi)
            trajs.append(pose)
    # for theta,phi in [(0.,0.), (0, np.pi)]:
    #     pose = get_pose(theta,phi)
    #     trajs.append(pose)
    target = np.array([0,0,2.0])
    point_up = np.array([0, 0, 13.5])
    pose_up = pose_point_to(loc=point_up*scale, target=target*scale)
    point_down = np.array([0, 0, -9])
    pose_down = pose_point_to(loc=point_down*scale, target=target*scale)
    trajs += [pose_up, pose_down]
    return trajs

def hubble_init(scale=1., n=5):
    trajs = []
    center = pose_point_to(loc=np.array([1.7, 1.7, 0.35])*6*scale, target=[0,0,0.35*6*scale])
    if n!=3:
        trajs.append(center)
    else:
        n=4
    d = 0.5
    d *= 6*scale
    for i in range(n-1):
        theta = np.pi*2/(n-1)*i
        # breakpoint()
        loc = center @ np.array([d*np.cos(theta), d*np.sin(theta), 0, 1])
        loc = loc[:3]

        z = -0.15 + i*1/(n-2) 
        target = np.array([0,0,z])*6*scale

        pose = pose_point_to(loc=loc, target=target)
        trajs.append(pose)
    return trajs

def lego_eval(scale=1.):
    def get_pose(theta, phi):
        target = np.array([0,0,0.35])
        point = np.array([1.5, 2, 1.7])*np.array([np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)]) + target
        
        pose = pose_point_to(loc=point*scale, target=target*scale)
        return pose
    trajs = []
    n1, n2 = 3, 6 #6, 8 # 3, 6
    altitude_list = np.linspace(0, np.pi/2, n1+1)[1:]
    
    for phi,n2 in zip(altitude_list, [4, 6, 9]):
        longitude_list = np.linspace(0, np.pi*2, n2+1)[:-1]
        for theta in  longitude_list:
            pose = get_pose(theta,phi)
            trajs.append(pose)
    # for theta,phi in [(0.,0.), (0, np.pi)]:
    #     pose = get_pose(theta,phi)
    #     trajs.append(pose)
    target = np.array([0,0,0.35])
    point_up = np.array([0, 0, 1.7])
    pose_up = pose_point_to(loc=point_up*scale, target=target*scale)
    # point_down = np.array([0, 0, -9])
    # pose_down = pose_point_to(loc=point_down*scale, target=target*scale)
    trajs += [pose_up]
    return trajs

def lego_init(scale=1., n=5):
    trajs = []
    center = pose_point_to(loc=np.array([0., 2.2, 0.35])*scale, target=[0,0,0.35*scale])
    if n!=3:
        trajs.append(center)
    else:
        n=4
    d = 0.3
    d *= scale
    for i in range(n-1):
        theta = np.pi*2/(n-1)*i
        # breakpoint()
        loc = center @ np.array([d*np.cos(theta), d*np.sin(theta), 0, 1])
        loc = loc[:3]

        z = -0.3 + i*0.9/(n-2) 
        target = np.array([0,0,z])*scale

        pose = pose_point_to(loc=loc, target=target)
        trajs.append(pose)
    return trajs

def hotdog_init(scale = 1., n=3, use_center=False):
    trajs = []
    center = pose_point_to(loc=np.array([0.0, 0., 1.2])*scale, target=[0,0,0.])
    if use_center:
        trajs.append(center)
    else:
        n+=1
    # if n!=3:
    #     trajs.append(center)
    # else:
    #     n=4
    d = 1.5
    d *= scale
    for i in range(n-1):
        theta = np.pi*2/(n-1)*i
        # breakpoint()
        loc = center @ np.array([d*np.cos(theta), d*np.sin(theta), 0, 1])
        loc = loc[:3]

        z = -0.15 + i*0.3/(n-2) 
        target = np.array([0,0,z])*scale

        pose = pose_point_to(loc=loc, target=target)
        trajs.append(pose)
    return trajs

def hotdog_eval(scale=1.):
    def get_pose(theta, phi):
        target = np.array([0,0,0.0])
        z0 =  np.array([0,-0.088056, 0.4])
        point = np.array([1.6, 1.6, 1.3])*np.array([np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)]) + z0
        
        pose = pose_point_to(loc=point*scale, target=target*scale)
        return pose
    trajs = []
    n1, n2 = 3, 6 #6, 8 # 3, 6
    altitude_list = np.linspace(0, np.pi/2, n1+1)[1:]
    
    for phi,n2 in zip(altitude_list, [4, 6, 9]):
        longitude_list = np.linspace(0, np.pi*2, n2+1)[:-1]
        for theta in  longitude_list:
            pose = get_pose(theta,phi)
            trajs.append(pose)
    # for theta,phi in [(0.,0.), (0, np.pi)]:
    #     pose = get_pose(theta,phi)
    #     trajs.append(pose)
    target = np.array([0,0,0.35])
    point_up = np.array([0, 0, 1.7])
    pose_up = pose_point_to(loc=point_up*scale, target=target*scale)
    # point_down = np.array([0, 0, -9])
    # pose_down = pose_point_to(loc=point_down*scale, target=target*scale)
    trajs += [pose_up]
    return trajs

def hotdog_full(Tmax=400, scale=1., pxpy=True):
    trajs=  []
    for _ in range(Tmax):
        if pxpy:
            theta, phi = np.random.rand()*1*np.pi/2 -np.pi/4, (np.random.rand()*1/4 + 1/4.)*np.pi

            
        else:
            theta, phi = np.random.rand()*np.pi*2, np.random.rand()*np.pi/2
        point = np.array([1.5, 1.5, 1.7])*np.array([np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)]) + np.array([0,0,0.])
        t_x = np.random.rand()-0.5
        t_y = np.random.rand()-0.5
        t_z = np.random.rand()*0.5
        target = np.array([t_x,t_y,t_z])
        pose = pose_point_to(loc=point*scale, target=target*scale, up=np.random.rand(3))
        # pose[:2, 3] = pose[:2, 3]/2
        trajs.append(pose)
    return trajs

def room_full(Tmax=400, scale=1.):
    trajs=  []
    for _ in range(Tmax):
        theta, phi = np.random.rand()*np.pi*2, np.random.rand()*np.pi
        point = 2*np.array([5, 5, 7.5])*np.array([np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)]) + np.array([0,0,3.])
        t_x = np.random.rand()-0.5
        t_y = np.random.rand()-0.5
        t_z = np.random.rand()*3.5-1.
        target = np.array([t_x,t_y,t_z])*2
        pose = pose_point_to(loc=point*scale, target=target*scale, up=np.random.rand(3))
        trajs.append(pose)
    return trajs

def room_eval(scale=1.):
    """
    Create evaluation dataset for room, 5 images in room 1 remaining in room 2
    """
    trajs = []
    radius = 4
    num_poses = 5
    for i in range(num_poses):
        theta = 2 * np.pi * i / num_poses  
        x = radius * np.cos(theta)  
        y = radius * np.sin(theta)  

        translation = np.array([[1, 0, 0, x],
                                [0, 1, 0, y],
                                [0, 0, 1, 4],
                                [0, 0, 0, 1]])
        
        rotation = np.array([[np.cos(theta), -np.sin(theta), 0, 0],
                             [np.sin(theta), np.cos(theta), 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])
        
        pose = np.dot(translation, rotation)

        # Append the pose to the list
        trajs.append(pose)
    radius = 3
    num_poses = 10
    for i in range(num_poses):
        theta = 2 * np.pi * i / num_poses  
        x = radius * np.cos(theta) - 8 
        y = radius * np.sin(theta) - 1

        translation = np.array([[1, 0, 0, x],
                                [0, 1, 0, y],
                                [0, 0, 1, 4],
                                [0, 0, 0, 1]])
        
        rotation = np.array([[np.cos(theta), -np.sin(theta), 0, 0],
                             [np.sin(theta), np.cos(theta), 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])
        
        pose = np.dot(translation, rotation)

        # Append the pose to the list
        trajs.append(pose)
    num_poses = 5
    for i in range(num_poses):
        theta = 2 * np.pi * i / num_poses  
        x = radius * np.cos(theta) - 3
        y = radius * np.sin(theta) + 2 

        translation = np.array([[1, 0, 0, x],
                                [0, 1, 0, y],
                                [0, 0, 1, 4],
                                [0, 0, 0, 1]])
        
        rotation = np.array([[np.cos(theta), -np.sin(theta), 0, 0],
                             [np.sin(theta), np.cos(theta), 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])
        
        pose = np.dot(translation, rotation)

        # Append the pose to the list
        trajs.append(pose)
    return trajs

def room_init(scale=1., n=5):
    trajs = []
    center = pose_point_to(loc=np.array([0.7, 2.0, 2.35])*6*scale, target=[-2,2,0.35*6*scale])
    if n!=3:
        trajs.append(center)
    else:
        n=4
    d = 2
    d *= 6*scale
    for i in range(n-1):
        theta = np.pi*2/(n-1)*i
        loc = center @ np.array([d*np.cos(theta), d*np.sin(theta), 0, 1])
        loc = loc[:3]

        z = -0.15 + i*1/(n-2) 
        target = np.array([0,0,z])*6*scale

        pose = pose_point_to(loc=loc, target=target)
        trajs.append(pose)
    return trajs

def room_part(scale=1., n=None):
    file_path = 'data/room/part/transforms.json'

    assert os.path.exists(file_path), "File not found: {}".format(file_path)

    trajs = get_transforms(path=file_path)

    if scale != 1.:
        for i, traj in enumerate(trajs):
            trajs[i][:3, 3] *= scale

    return trajs
