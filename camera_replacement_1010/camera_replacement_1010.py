# %% [markdown]
# # Camera Position

# %% [markdown]
# xvfb-run --auto-servernum --server-args="-screen 0 640x480x24" /unity_vol/linux_exec.v2.3.0.x86_64 -batchmode -http-port=8081

# %%

import IPython.display
from utils_demo import *
from utils_view import *
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
np.set_printoptions(suppress=True)

from unity_simulator.comm_unity import UnityCommunication
from unity_simulator import utils_viz

comm = UnityCommunication(port = '8081')

# %% [markdown]
# ## Whole scene

# %%
comm.procedural_generation(90)
_, graph = comm.environment_graph()
comm.remove_terrain()
IPython.display.display(display_whole_scene(comm, field_view = 40))
room = find_nodes(graph, category = 'Rooms')[0]

# %% [markdown]
# ## Top view

# %%
IPython.display.display(display_top_room(comm, room))

# %% [markdown]
# ## CCTV View

# %%
display_cctv_room(comm, room, nrows = 2)

# %% [markdown]
# ## Closeup

# %%
def get_square_small_table(comm, r, o):
    """ nightstand, tvtable, coffeetable """
    o_pos = [0, 0, 0]
    sign_x = 1 if r['center'][0] - o['center'][0] > 0 else -1
    sign_z = 1 if r['center'][2] - o['center'][2] > 0 else -1            
    o_pos[0] = o['center'][0] + 2 * sign_x
    o_pos[1] = o['center'][1] + 0.8
    o_pos[2] = o['center'][2] + 2 * sign_z
    comm.add_camera(position=o_pos, rotation=[10 ,180 * max(sign_z, 0) + 35 * sign_x * sign_z,0], field_view= 50)
    o_pos[0] = o['center'][0] + 1 * sign_x 
    o_pos[1] = o['center'][1] + 1
    o_pos[2] = o['center'][2] + 1 * sign_z
    comm.add_camera(position=o_pos, rotation=[30 ,180 * max(sign_z, 0) + 35 * sign_x * sign_z,0], field_view= 50)
    o_pos[0] = o['center'][0] + o['size'][0] * sign_x * 0.5
    o_pos[1] = o['center'][1] + 1.5
    o_pos[2] = o['center'][2] + o['size'][2] * sign_z * 0.5
    comm.add_camera(position=o_pos, rotation=[90 - o['size'][0] * 20 ,180 * max(sign_z, 0) + 30 * sign_x * sign_z + o['size'][0] * 12, 0], field_view= 50)
    c = comm.camera_count()[1] - 1
    return c

def get_square_large_table(comm, r, o):
    """ coffeetable """
    o_pos = [0, 0, 0]
    sign_x = 1 if r['center'][0] - o['center'][0] > 0 else -1
    sign_z = 1 if r['center'][2] - o['center'][2] > 0 else -1
    o_pos[0] = o['center'][0] - o['size'][0] * sign_x * 1.3
    o_pos[1] = o['center'][1] + 1.5
    o_pos[2] = o['center'][2] 
    comm.add_camera(position=o_pos, rotation=[38 ,180 * max(sign_x, 0) - 90 + 5 * sign_x * sign_z,0], field_view= 45)
    o_pos[0] = o['center'][0] 
    o_pos[1] = o['center'][1] + 1.5
    o_pos[2] = o['center'][2] + o['size'][2] * sign_z * 0.3
    comm.add_camera(position=o_pos, rotation=[90 - o['size'][0] * 10 ,180 * max(sign_z, 0), 0], field_view= 50)
    o_pos[0] = o['center'][0] + o['size'][0] * sign_x * 1.3
    o_pos[1] = o['center'][1] + 1.5
    o_pos[2] = o['center'][2]
    comm.add_camera(position=o_pos, rotation=[38 ,180 * max(sign_x, 0) + 90 - 5 * sign_x * sign_z,0], field_view= 45)
    c = comm.camera_count()[1] - 1
    return c

# %%
def get_square_kitchencounter(comm, r, o):            
    o_pos = [0, 0, 0]
    sign_x = 1 if r['center'][0] - o['center'][0] > 0 else -1
    sign_z = 1 if r['center'][2] - o['center'][2] > 0 else -1
    o_pos[0] = o['center'][0] + o['size'][0] * sign_x * 0.5
    o_pos[1] = o['center'][1] + 1
    o_pos[2] = o['center'][2] + o['size'][2] * sign_z * 0.8
    comm.add_camera(position=o_pos, rotation=[0 ,180 * max(sign_z, 0)  + 36 * sign_x * sign_z,0], field_view= 45)
    o_pos[0] = o['center'][0] 
    o_pos[1] = o['center'][1] + 1
    o_pos[2] = o['center'][2] + o['size'][2] * sign_z * 0.3
    comm.add_camera(position=o_pos, rotation=[0, 180 * max(sign_z, 0), 0], field_view= 45)
    o_pos[0] = o['center'][0] + o['size'][0] * sign_x * 0.3
    o_pos[1] = o['center'][1] + 1
    o_pos[2] = o['center'][2] + o['size'][2] * sign_z * 0
    comm.add_camera(position=o_pos, rotation=[0 ,180 * max(sign_z, 0) + 90 - 0 * sign_x * sign_z,0], field_view= 45)
    c = comm.camera_count()[1] - 1
    return c

# %%
def get_rectangular_sofa(comm, r, o):            
    o_pos = [0, 0, 0]
    sign_x = 1 if r['center'][0] - o['center'][0] > 0 else -1
    sign_z = 1 if r['center'][2] - o['center'][2] > 0 else -1
    if o['size'][0] < o['size'][2]:
        o_pos[0] = o['center'][0] + o['size'][0] * sign_x * 2.2
        o_pos[1] = o['center'][1] + 1
        o_pos[2] = o['center'][2] + o['size'][2] * sign_z * 2
        comm.add_camera(position=o_pos, rotation=[10 ,180 * max(sign_z, 0) + (90 - 25) * sign_x * sign_z,0], field_view= 50)        
        o_pos[0] = o['center'][0] + o['size'][0] * sign_x * 2.2
        o_pos[1] = o['center'][1] + 0.8
        o_pos[2] = o['center'][2] 
        comm.add_camera(position=o_pos, rotation=[15 ,180 * max(sign_z, 0) + 90 * sign_x * sign_z, 0], field_view= 50)
        o_pos[0] = o['center'][0] + o['size'][0] * sign_x * 1
        o_pos[1] = o['center'][1] + 1.4
        o_pos[2] = o['center'][2] 
        comm.add_camera(position=o_pos, rotation=[90 - 40 ,180 * max(sign_z, 0) + 90 * sign_x * sign_z, 0], field_view= 50)
    else:
        o_pos[0] = o['center'][0] + o['size'][2] * sign_x * 2
        o_pos[1] = o['center'][1] + 1
        o_pos[2] = o['center'][2] + o['size'][2] * sign_z * 2.2
        comm.add_camera(position=o_pos, rotation=[10 ,180 * max(sign_z, 0) + 75 * sign_x * sign_z,0], field_view= 50)      
        o_pos[0] = o['center'][0] 
        o_pos[1] = o['center'][1] + 0.8
        o_pos[2] = o['center'][2] + o['size'][2] * sign_z * 2.2
        comm.add_camera(position=o_pos, rotation=[15 ,180 * max(sign_z, 0) + 0 * sign_x * sign_z, 0], field_view= 50)
        o_pos[0] = o['center'][0] 
        o_pos[1] = o['center'][1] + 1.4
        o_pos[2] = o['center'][2] + o['size'][2] * sign_z * 1
        comm.add_camera(position=o_pos, rotation=[90 - 40 , 180 * max(sign_z, 0) + 0 * sign_x * sign_z, 0], field_view= 50)
    c = comm.camera_count()[1] - 1
    return c

# %%
def get_rectangular_tvstand(comm, r, o):            
    o_pos = [0, 0, 0]
    sign_x = 1 if r['center'][0] - o['center'][0] > 0 else -1
    sign_z = 1 if r['center'][2] - o['center'][2] > 0 else -1
    if o['size'][0] > o['size'][2]:
        o_pos[0] = o['center'][0] + o['size'][0] * sign_x * 0.5
        o_pos[1] = o['center'][1] + 1
        o_pos[2] = o['center'][2] + o['size'][0] * sign_z * 1.5
        comm.add_camera(position=o_pos, rotation=[10 ,180 * max(sign_z, 0) + 75 * sign_x * sign_z,0], field_view= 50)      
        o_pos[0] = o['center'][0] 
        o_pos[1] = o['center'][1] + 0.8
        o_pos[2] = o['center'][2] + o['size'][0] * sign_z * 1
        comm.add_camera(position=o_pos, rotation=[15 ,180 * max(sign_z, 0) + 0 * sign_x * sign_z, 0], field_view= 50)
        o_pos[0] = o['center'][0] 
        o_pos[1] = o['center'][1] + 1.4
        o_pos[2] = o['center'][2] + o['size'][0] * sign_z * 0.6
        comm.add_camera(position=o_pos, rotation=[40 , 180 * max(sign_z, 0) + 0 * sign_x * sign_z, 0], field_view= 50)
    else:
        o_pos[0] = o['center'][0] + o['size'][2] * sign_x * 1.5
        o_pos[1] = o['center'][1] + 1
        o_pos[2] = o['center'][2] + o['size'][2] * sign_z * 0.5
        comm.add_camera(position=o_pos, rotation=[10 ,180 * max(sign_x, 0) + (90 - 25) * sign_x * sign_z,0], field_view= 50)        
        o_pos[0] = o['center'][0] + o['size'][2] * sign_x * 1
        o_pos[1] = o['center'][1] + 0.8
        o_pos[2] = o['center'][2] 
        comm.add_camera(position=o_pos, rotation=[15 ,180 * max(sign_x, 0) + 90 * sign_x * sign_z, 0], field_view= 50)
        o_pos[0] = o['center'][0] + o['size'][2] * sign_x * 0.6
        o_pos[1] = o['center'][1] + 1.4
        o_pos[2] = o['center'][2] 
        comm.add_camera(position=o_pos, rotation=[40 ,180 * max(sign_x, 0) + 90 * sign_x * sign_z, 0], field_view= 50)

        
    c = comm.camera_count()[1] - 1
    return c

# %%
def get_rectangular_desk(comm, r, o):            
    o_pos = [0, 0, 0]
    sign_x = 1 if r['center'][0] - o['center'][0] > 0 else -1
    sign_z = 1 if r['center'][2] - o['center'][2] > 0 else -1
    if o['size'][0] - o['size'][2] > 0:
        o_pos[0] = o['center'][0] - o['size'][0] * sign_x * 0.5
        o_pos[1] = o['center'][1] + 1.5
        o_pos[2] = o['center'][2] + o['size'][0] * sign_z * 1.2
        comm.add_camera(position=o_pos, rotation=[30 ,180 * max(sign_z, 0) + 90 + 65 * sign_x * sign_z,0], field_view= 50)
        o_pos[0] = o['center'][0] 
        o_pos[1] = o['center'][1] + 1.8
        o_pos[2] = o['center'][2] + o['size'][0] * sign_z * 0.4
        comm.add_camera(position=o_pos, rotation=[90 - o['size'][0] * 15 ,180 * max(sign_z, 0), 0], field_view= 60)
        o_pos[0] = o['center'][0] + o['size'][0] * sign_x * 1.3
        o_pos[1] = o['center'][1] + 1.4
        o_pos[2] = o['center'][2]
        comm.add_camera(position=o_pos, rotation=[20 ,180 * max(sign_x, 0) + 90 - 5 * sign_x * sign_z ,0], field_view= 40)
    else:
        o_pos[0] = o['center'][0] + sign_x * o['size'][2] * 1.2
        o_pos[1] = o['center'][1] + 1.5
        o_pos[2] = o['center'][2] - sign_z * o['size'][2] * 0.5
        comm.add_camera(position=o_pos, rotation=[30 , + 180 * max(sign_x, 0) + 25 * sign_x * sign_z,0], field_view= 50)
        o_pos[0] = o['center'][0] + o['size'][0] * sign_x * 0.4
        o_pos[1] = o['center'][1] + 1.8
        o_pos[2] = o['center'][2] 
        comm.add_camera(position=o_pos, rotation=[90 - o['size'][2] * 15 ,90 + 180 * max(sign_x, 0), 0], field_view= 60)
        o_pos[0] = o['center'][0] 
        o_pos[1] = o['center'][1] + 1.4
        o_pos[2] = o['center'][2] + o['size'][2] * sign_z * 1.3
        comm.add_camera(position=o_pos, rotation=[20 ,180 * max(sign_z, 0)  + 90 - 5 * sign_x * sign_z,0], field_view= 40)
    c = comm.camera_count()[1] - 1
    return c

# %%
def get_large_furniture(comm, r, o):            
    o_pos = [0, 0, 0]
    sign_x = 1 if r['center'][0] - o['center'][0] > 0 else -1
    sign_z = 1 if r['center'][2] - o['center'][2] > 0 else -1
    if o['size'][0] - o['size'][2] > 0:
        o_pos = o['center'].copy()
        o_pos[0] = o['center'][0] + 0.5 * sign_x
        o_pos[1] = 1.7
        o_pos[2] = o['center'][2] + 3 * sign_z
        comm.add_camera(position=o_pos, rotation=[13, 180 * max(sign_z, 0) + 10 * sign_x * sign_z,0], field_view= 50)
        o_pos[0] = o['center'][0] 
        o_pos[1] = o['center'][1] + 0.5
        o_pos[2] = o['center'][2] + sign_z * (1.2 + o['size'][0] * 0.5)
        comm.add_camera(position=o_pos, rotation=[8 ,180 * max(sign_z, 0) , 0], field_view= 50)
        o_pos[0] = o['center'][0] 
        o_pos[1] = o['center'][1] - 0.5
        o_pos[2] = o['center'][2] + sign_z * (1.2 + o['size'][0] * 0.5)
        comm.add_camera(position=o_pos, rotation=[0 ,180 * max(sign_z, 0) ,0], field_view= 40)
    else:
        o_pos = o['center'].copy()
        o_pos[0] += 3 * sign_x
        o_pos[1] = 1.7
        o_pos[2] += 0.5 * sign_z
        comm.add_camera(position=o_pos, rotation=[13, 90 + 180 * max(sign_x, 0) + 10 * sign_x * sign_z,0], field_view= 50)
        o_pos[0] = o['center'][0] + sign_x * (1.2 + o['size'][2] * 0.5)
        o_pos[1] = o['center'][1] + 0.5
        o_pos[2] = o['center'][2] 
        comm.add_camera(position=o_pos, rotation=[8 ,90 + 180 * max(sign_x, 0), 0], field_view= 50)
        o_pos[0] = o['center'][0] + sign_x * (1.2 + o['size'][2] * 0.5)
        o_pos[1] = o['center'][1] - 0.5
        o_pos[2] = o['center'][2]
        comm.add_camera(position=o_pos, rotation=[0 , 90 + 180 * max(sign_x, 0),0], field_view= 40)
    c = comm.camera_count()[1] - 1
    return c

# %%
def display_closeup_room(comm, room, objs_list = objs_list, nrows = 2):
    _, graph = comm.environment_graph()
    obj_in_room = find_edges_to(graph, room['id'])
    objs = [obj for _, obj in obj_in_room if obj['class_name'] in objs_list]
    if len(objs) == 0:
        return []
    else :
        r = room['bounding_box']
        cameras = []
        for obj in objs[:]:
            o = obj['bounding_box']
            if o['size'][1] > 1.5:
                # closet, bookshelf
                c = get_large_furniture(comm, r, o)
            else :
                if abs(o['size'][0] - o['size'][2]) < 0.2:
                    # square furniture
                    if o['size'][2] > 2: 
                        # kitchencounter
                        c = get_square_kitchencounter(comm, r, o)
                    elif o['size'][2] > 1: 
                        # coffetable
                        c = get_square_large_table(comm, r, o)
                    else : 
                        # nightstand
                        c = get_square_small_table(comm, r, o)
                else:
                    # rectangular furniture
                    if abs(o['size'][0] - o['size'][2]) >= 1: 
                        if o['size'][0] < 1:
                            # tvstand
                            c = get_rectangular_tvstand(comm, r, o)

                        else:
                            # rectangular sofa
                            c = get_rectangular_sofa(comm, r, o)
                    else: 
                        # desk, kitchentable
                        c = get_rectangular_desk(comm, r, o)
            cameras.append(c - 2)
            cameras.append(c - 1)
            cameras.append(c - 0)
            
        return display_comparison_cameras(comm, cameras, view = 3, nrows = nrows)
    
display_closeup_room(comm, find_nodes(graph, category = 'Rooms')[3])

# %% [markdown]
# ## New closeup

# %%
get_closeup_room(comm, room, view = 3, nrows = 1)

# %%
def display_comparison_img(images_old, view = 3 ,nrows=1):
    images = [x for x in images_old]
    h, w, _ = images[0].shape
    ncols = np.ceil(len(images) / nrows / view).astype(int)

    missing = ncols * view * nrows - len(images)
    for _ in range(missing):
        images.append(np.zeros((h, w, 3)).astype(np.uint8))
    img_final = []
    for it_r in range(nrows):
        init_ind = it_r * ncols * view
        end_ind = init_ind + ncols * view
        for it_v in range(view):
            images_take = [images[it] for it in range(init_ind + it_v, end_ind, view)]
            img_final.append(np.concatenate(images_take, 1))
    img_final = np.concatenate(img_final, 0)
    return PIL.Image.fromarray(img_final[:,:,::-1])

def get_scene_cameras(comm, ids, mode='normal'):
    _, ncameras = comm.camera_count()
    cameras_select = list(range(ncameras))
    cameras_select = [cameras_select[x] for x in ids]
    (ok_img, imgs) = comm.camera_image(cameras_select, mode=mode, image_width=640, image_height=360)
    return imgs

def display_comparison_cameras(comm, ids, view = 2, nrows=1, mode='normal'):
    imgs = get_scene_cameras(comm, ids, mode)
    return display_comparison_img(imgs, view = view, nrows=nrows)

# %%
def get_closeup_obj(comm, o, camera_args, sign_x, sign_z):            
    o_pos = o['center'].copy()
    x, h, z, pitch, yaw, field_view = camera_args
    if o['size'][0] - o['size'][2] > 0:
        l = o['size'][0]
        o_pos[0] += l * sign_x * x
        o_pos[1] += h
        o_pos[2] += l * sign_z * z
        comm.add_camera(position=o_pos, rotation=[pitch ,180 * max(sign_z, 0) + yaw * sign_x * sign_z,0], field_view= field_view)
    else:
        l = o['size'][2]
        o_pos[0] += l * sign_x * z
        o_pos[1] += h
        o_pos[2] += l * sign_z * x
        comm.add_camera(position=o_pos, rotation=[pitch ,180 * max(sign_x, 0) - yaw * sign_x * sign_z + 90, 0], field_view= field_view)

# %%
def get_closeups_obj(comm, r, obj, cameras,):
    o = obj['bounding_box']
    class_name = obj['class_name']
    if class_name in ['closet', 'bookshelf'] and max(o['size'][0], o['size'][2]) < 1.5:
        class_name = obj['class_name'] + '_small'
    sign_x = 1 if r['center'][0] - o['center'][0] > 0 else -1
    sign_z = 1 if r['center'][2] - o['center'][2] > 0 else -1
    
    for _, view in obj_arg[class_name].items():
        get_closeup_obj(comm, o, view, sign_x, sign_z)
    c = comm.camera_count()[-1] - 1
    for i in range(2, -1, -1):
        cameras.append(c - i)
    return cameras

# %%
obj_arg = dict()
# x, h, z, pitch, yaw, field_view
obj_arg['desk'] = {'full_view' :[-0.5, 1.5, 1.2, 25, -25, 50],
                    'top_view' :[0, 2, 0.5, 60, 0, 70],
                    'side_view':[1.3, 1.4, 0, 20, 85, 50]}
obj_arg['kitchentable'] = {'full_view' :[-0.5, 1.5, 1.2, 25, -25, 50],
                    'top_view' :[0, 2, 0.5, 60, 0, 70],
                    'side_view':[1.3, 1.4, 0, 20, 85, 50]} 
obj_arg['sofa'] = {'full_view' :[-0.5, 1.5, 1.2, 25, -25, 50],
                    'top_view' :[0, 2, 0.5, 60, 0, 70],
                    'side_view':[1.3, 1.4, 0, 20, 85, 50]} 
obj_arg['tvstand'] = {'full_view' :[-0.5, 1.2, 1.2, 25, -25, 50],
                    'top_view' :[0, 1, 0.5, 30, 0, 70],
                    'side_view':[1.3, 1.4, 0, 20, 85, 50]} 
obj_arg['coffeetable'] = {'full_view' :[-0.5, 1.5, 1.6, 30, -25, 50],
                    'top_view' :[0, 2, 0.5, 60, 0, 70],
                    'side_view':[1.6, 1.2, 0, 20, 85, 50]} 
obj_arg['bookshelf'] = {'full_view' :[0.8, 0, 0.8, 0, 25, 90],
                    'top_view' :[0, 0.8, 0.5, 10, 0, 75],
                    'side_view':[0, -0.2, 0.5, 10, 0, 75]} 
obj_arg['closet'] = {'full_view' :[0.8, 0, 0.8, 0, 25, 90],
                    'top_view' :[0, 0.8, 0.5, 10, 0, 75],
                    'side_view':[0, -0.2, 0.5, 10, 0, 75]} 
obj_arg['nightstand'] = {'full_view' :[-0.5, 1.2, 1.2, 25, -25, 50],
                    'top_view' :[0, 1.3, 0.5, 70, 0, 70],
                    'side_view':[1.7, 1.4, 0, 30, 85, 50]} 
obj_arg['kitchencounter'] = {'full_view' :[-0.5, 1, 0.5, 0, -15, 90], 
                    'top_view' :[0, 1., 0.5, 0, 0, 70],
                    'side_view':[0.5, 0.8, 0.5, 0, 15, 90]} 
obj_arg['bathroomcounter'] = {'full_view' :[-0.4, 0.8, 0.4, 0, -15, 90], 
                    'top_view' :[0, 1., 0.5, 0, 0, 70],
                    'side_view':[0.4, 0.8, 0.4, 0, 15, 90]} 
obj_arg['stall'] = {'full_view' :[-0.2, 0.2, 0.5, 0, -15, 90], 
                    'top_view' :[0, 1, 0.2, 70, 0, 50],
                    'side_view':[0.3, 0.2, 0.5, 0, 15, 90]}
obj_arg['bookshelf_small'] = {'full_view' :[0.1, 0.2, 0.8, 0, 0, 120],
                    'top_view' :[0, 0.8, 0.5, 10, 0, 90],
                    'side_view':[0, -0.2, 0.5, 15, 0, 90]}  
obj_arg['closet_small'] = {'full_view' :[0.1, 0.2, 0.8, 0, 0, 120],
                    'top_view' :[0, 0.8, 0.5, 5, 0, 100],
                    'side_view':[0, -0.4, 0.5, 5, 0, 100]}  
objs_list = ["bench",
            "coffeetable",
            "desk",
            'bathroomcounter',
            "stall",
            "kitchencounter",
            "kitchentable",
            "nightstand",
            "sofa",
            "tvstand",
            'closet',
            'bookshelf']

# %%
def get_closeup_room(comm, room, view = 3, nrows = 1):

    cameras = []
    obj_in_room = find_edges_to(graph, room['id'])
    objs = [obj for _, obj in obj_in_room if (obj['class_name']== 'kitchencounter' and min(obj['bounding_box']['size']) > 0.6) or obj['class_name'] in objs_list]
    if len(objs) == 0:
        cameras = []
    else :
        r = room['bounding_box']
        for obj in objs[:]:
            cameras = get_closeups_obj(comm, r, obj, cameras,)
    return display_comparison_cameras(comm, cameras, view = view, nrows = nrows)


