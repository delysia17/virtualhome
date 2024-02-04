import PIL
from utils_demo import *
import numpy as np

sys.path.append('/DATA/disk1/chenziyi/virtualhome/virtualhome/simulation')
from unity_simulator.comm_unity import UnityCommunication
from unity_simulator import utils_viz

objs_list = ["coffeetable",
            "desk",
            'bathroomcounter',
            "stall",
            # "kitchencounter",
            "kitchentable",
            # "nightstand",
            "sofa",
            "tvstand",
            # 'closet',
            # 'bookshelf',
            'cabinet']


obj_arg = {'desk': {'full_view': [-0.5, 1.5, 1.2, 25, -25, 50],
  'top_view': [0, 2, 0.5, 60, 0, 70],
  'side_view': [1.3, 1.4, 0, 20, 85, 50]},
  'cabinet': {'full_view': [-0.5, 1.5, 1.2, 25, -25, 50],
  'top_view': [0, 2, 0.5, 60, 0, 70],
  'side_view': [1.3, 1.4, 0, 20, 85, 50]},
 'kitchentable': {'full_view': [-0.5, 1.5, 1.2, 25, -25, 50],
  'top_view': [0, 2, 0.5, 60, 0, 70],
  'side_view': [1.3, 1.4, 0, 20, 85, 50]},
 'sofa': {'full_view': [-0.5, 1.5, 1.2, 25, -25, 50],
  'top_view': [0, 2, 0.5, 60, 0, 70],
  'side_view': [1.3, 1.4, 0, 20, 85, 50]},
 'tvstand': {'full_view': [-0.5, 1.2, 1.2, 25, -25, 50],
  'top_view': [0, 1, 0.5, 30, 0, 70],
  'side_view': [1.3, 1.4, 0, 20, 85, 50]},
 'coffeetable': {'full_view': [-0.5, 1.5, 1.6, 30, -25, 50],
  'top_view': [0, 2, 0.5, 60, 0, 70],
  'side_view': [1.6, 1.2, 0, 20, 85, 50]},
 'bookshelf': {'full_view': [0.8, 0, 0.8, 0, 25, 90],
  'top_view': [0, 0.8, 0.5, 10, 0, 75],
  'side_view': [0, -0.2, 0.5, 10, 0, 75]},
 'closet': {'full_view': [0.8, 0, 0.8, 0, 25, 90],
  'top_view': [0, 0.8, 0.5, 10, 0, 75],
  'side_view': [0, -0.2, 0.5, 10, 0, 75]},
 'nightstand': {'full_view': [-0.5, 1.2, 1.2, 25, -25, 50],
  'top_view': [0, 1.3, 0.5, 70, 0, 70],
  'side_view': [1.7, 1.4, 0, 30, 85, 50]},
 'kitchencounter': {'full_view': [-0.5, 1, 0.5, 0, -15, 90],
  'top_view': [0, 1.0, 0.5, 0, 0, 70],
  'side_view': [0.5, 0.8, 0.5, 0, 15, 90]},
 'bathroomcounter': {'full_view': [-0.4, 0.8, 0.4, 0, -15, 90],
  'top_view': [0, 1.0, 0.5, 0, 0, 70],
  'side_view': [0.4, 0.8, 0.4, 0, 15, 90]},
 'stall': {'full_view': [-0.2, 0.2, 0.5, 0, -15, 90],
  'top_view': [0, 1, 0.2, 70, 0, 50],
  'side_view': [0.3, 0.2, 0.5, 0, 15, 90]},
 'bookshelf_small': {'full_view': [0.1, 0.2, 0.8, 0, 0, 120],
  'top_view': [0, 0.8, 0.5, 10, 0, 90],
  'side_view': [0, -0.2, 0.5, 15, 0, 90]},
 'closet_small': {'full_view': [0.1, 0.2, 0.8, 0, 0, 120],
  'top_view': [0, 0.8, 0.5, 5, 0, 100],
  'side_view': [0, -0.4, 0.5, 5, 0, 100]}}

## Utils adding
def cctv_rectangular(comm, r, camera_args, nrows = 2):
    """     Placement of CCTV in rectangular room
        right >> [x + size * scale, y, z],
        left >> [x - size * scale, y, z]
        upper >> [x, y, z + size * scale]
        lower >> [x, y, z - size * scale]] """
    x, h, z, pitch, yaw, field_view = camera_args
    cctv_dict = {}

    upper_right_pos  = [r['center'][0] + r['size'][0] * x, h, r['center'][2] + r['size'][2] * z]
    upper_right_rot = [pitch, 180 + yaw, 0]
    upper_left_pos  = [r['center'][0] - r['size'][0] * x, h, r['center'][2] + r['size'][2] * z]
    upper_left_rot = [pitch, 180 - yaw, 0]
    lower_left_pos  = [r['center'][0] - r['size'][0] * x, h, r['center'][2] - r['size'][2] * z]
    lower_left_rot = [pitch, yaw, 0]
    lower_right_pos  = [r['center'][0] + r['size'][0] * x, h, r['center'][2] - r['size'][2] * z]
    lower_right_rot = [pitch, - yaw, 0]

    comm.add_camera(position=upper_right_pos, rotation = upper_right_rot, field_view = field_view)
    cctv_dict['upper_right_corner'] = comm.camera_count()[-1] - 1
    comm.add_camera(position=upper_left_pos, rotation = upper_left_rot, field_view = field_view)
    cctv_dict['upper_left_corner'] = comm.camera_count()[-1] - 1
    comm.add_camera(position=lower_left_pos, rotation = lower_left_rot, field_view = field_view)
    cctv_dict['lower_left_corner'] = comm.camera_count()[-1] - 1
    comm.add_camera(position=lower_right_pos, rotation = lower_right_rot, field_view = field_view)
    cctv_dict['lower_right_corner'] = comm.camera_count()[-1] - 1

    return display_scene_cameras(comm, [ cctv_dict['lower_right_corner'], cctv_dict['lower_left_corner'], cctv_dict['upper_right_corner'], cctv_dict['upper_left_corner']], nrows = nrows) , cctv_dict

def cctv_square(comm, r, camera_args, nrows = 2):
    """     Placement of CCTV in square room
        right >> [x + size * scale, y, z],
        left >> [x - size * scale, y, z]
        upper >> [x, y, z + size * scale]
        lower >> [x, y, z - size * scale]] """
    x, h, z, pitch, _, field_view = camera_args
    cctv_dict = {}
    
    upper_pos  = [r['center'][0], h, r['center'][2] + r['size'][2] * z]
    upper_rot = [pitch, 180, 0]
    lower_pos  = [r['center'][0], h, r['center'][2] - r['size'][2] * z]
    lower_rot = [pitch, 0, 0]
    left_pos  = [r['center'][0] - r['size'][0] * x, h, r['center'][2]]
    left_rot = [pitch, 90, 0]
    right_pos  = [r['center'][0] + r['size'][0] * x, h, r['center'][2]]
    right_rot = [pitch, - 90, 0]

    comm.add_camera(position=upper_pos, rotation = upper_rot, field_view = field_view)
    cctv_dict['lower_corner'] = comm.camera_count()[-1] - 1
    comm.add_camera(position=left_pos, rotation = left_rot, field_view = field_view)
    cctv_dict['left_corner'] = comm.camera_count()[-1] - 1
    comm.add_camera(position=lower_pos, rotation = lower_rot, field_view = field_view)
    cctv_dict['upper_corner'] = comm.camera_count()[-1] - 1
    comm.add_camera(position=right_pos, rotation = right_rot, field_view = field_view)
    cctv_dict['right_corner'] = comm.camera_count()[-1] - 1
    
    return display_scene_cameras(comm, [ cctv_dict['lower_corner'], cctv_dict['left_corner'], cctv_dict['right_corner'], cctv_dict['upper_corner']], nrows = nrows) , cctv_dict

def display_cctv_room(comm, room, nrows = 2):
    r = room['bounding_box']
    args = {
        '8,8':[0.15, 1.5, 0.15, 10, 0, 75],
        '5,5':[0.3, 1.5, 0.3, 10, 0, 65],
        '8,5':[0.25, 1.5, 0.2, 10, 70, 65],
        '5,8':[0.2, 1.5, 0.25, 10, 20, 65],} # x, h, z, pitch, yaw, field_view
    # rectangular room
    if r['size'][0] != r['size'][2]: 
        if r['size'][0] < r['size'][2]:
            imgs, _ = cctv_rectangular(comm, r, args['5,8'], nrows)
        else:
            imgs, _ = cctv_rectangular(comm, r, args['8,5'], nrows)
    # square room
    elif r['size'][0] == r['size'][2]:
        if r['size'][0] == 8.0:
            imgs, _ = cctv_square(comm, r, args['8,8'], nrows)
        else :
            imgs, _ = cctv_square(comm, r, args['5,5'], nrows)
    return imgs

def display_whole_scene(comm, field_view = 40 ,norws = 1):
    _, graph = comm.environment_graph()
    rooms = find_nodes(graph, category = 'Rooms')
    r = [room['bounding_box']['center'] for room in rooms]
    r = np.array(r)
    average_center = (np.max(r, axis = 0) + np.min(r, axis = 0)) / 2
    pos = [average_center[0], 40, average_center[2]]
    comm.add_camera(position=pos, rotation = [90, 0, 0], field_view= field_view)
    c = comm.camera_count()[-1] - 1
    return display_scene_cameras(comm, [c], nrows = norws)

    
def display_top_room(comm, room, nrows = 1):
    r = room['bounding_box']
    if max(r['size']) < 6:
        h = 10
    elif max(r['size']) < 9 :
        h = 12
    else:
        h = 15
    pos = r['center'].copy()
    pos[1] = h
    comm.add_camera(position=pos, rotation = [90, 0, 0])
    c = comm.camera_count()[-1] - 1
    return display_scene_cameras(comm, [c], nrows = nrows)

def display_comparison_img(images_old, view = 3 ,nrows=1):
    images = [x for x in images_old]
    h, w, _ = images[0].shape
    ncols = np.ceil(len(images) / nrows / view).astype(int)

    missing = ncols * view * nrows - len(images)
    for _ in range(missing):
        white_image = np.ones((h, w, 3)).astype(np.uint8) * 255
        images.append(white_image)
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
    nrows = np.ceil(len(imgs) / 6).astype(int)
    return display_comparison_img(imgs, view = view, nrows=nrows)

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
        return 

def get_closeups_obj(comm, r, obj, cameras,):
    o = obj['bounding_box']
    class_name = obj['class_name']
    if class_name in ['closet', 'bookshelf'] and max(o['size'][0], o['size'][2]) < 1.5 :
        class_name = obj['class_name'] + '_small'
    sign_x = 1 if r['center'][0] - o['center'][0] > 0 else -1
    sign_z = 1 if r['center'][2] - o['center'][2] > 0 else -1
    
    for _, view in obj_arg[class_name].items():
        get_closeup_obj(comm, o, view, sign_x, sign_z)
    c = comm.camera_count()[-1] - 1
    for i in range(2, -1, -1):
        cameras.append(c - i)
    return cameras

def get_closeup_room(comm, room, view = 3, nrows = 1):
    _, graph = comm.environment_graph()
    cameras = []
    obj_in_room = find_edges_to(graph, room['id'])
    objs = [obj for _, obj in obj_in_room if (obj['class_name']== 'kitchencounter' and max(obj['bounding_box']['size']) > 1.1) or obj['class_name'] in objs_list]
    if len(objs) == 0:
        cameras = []
    else :
        r = room['bounding_box']
        for obj in objs[:]:
            cameras = get_closeups_obj(comm, r, obj, cameras,)
    return display_comparison_cameras(comm, cameras, view = view, nrows = nrows)


import json

def save_json(filename, data):
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

def load_json(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data