import numpy as np
import trimesh
import os
import cv2
import json
import re
from pat3d.preprocessing.obj_io import rewrite_obj_vertex_lines
from pat3d.preprocessing.utils import draw_colored_points_to_obj


def _absolute_size_categories(size_ratio):
    if not isinstance(size_ratio, dict):
        return set()
    raw_value = size_ratio.get('__absolute_size_categories__')
    if raw_value is None:
        return set()
    if isinstance(raw_value, str):
        return {raw_value}
    if isinstance(raw_value, (list, tuple, set)):
        return {str(value) for value in raw_value if str(value)}
    return set()

## put the duplicate objects into it's container --> random position 
def get_initial_layout_duplicate(args, input_folder, input_depth_folder, input_seg_folder, 
                       case_name, output_dir, front_num, size_ratio):

    ## load the point cloud data
    pc_path = f'{input_depth_folder}/{case_name}/{case_name}_point_cloud.npz'
    pc_data = load_pc_data(pc_path)

    ## load the segmented items
    seg_folder = f'{input_seg_folder}/{case_name}'
    seg_items_dict = load_seg_items(seg_folder, case_name)

    ## seg the point cloud
    pc_dict = split_point_cloud(pc_data, seg_items_dict)

    ## load the obj meshes
    mesh_dict = {}
    for obj_file in os.listdir(f'{input_folder}/{case_name}'):
        if obj_file.split('.')[-1] == 'obj':
            obj_name = obj_file.split('.')[0]
            obj_path = f'{input_folder}/{case_name}/{obj_file}'
            obj_v, textured_obj, obj_path = load_obj_info_texture(obj_path)
            mesh_dict[obj_name] = {
                'v': obj_v,
                'textured_obj': textured_obj,
                'obj_path': obj_path
            }


    ## compute the front mean position of the object
    front_avg_dict = {}
    for pc_name, pc in pc_dict.items():
        front_num = min(len(pc), front_num)
        if pc_name == 'bookshelf2':
            start_index = int(len(pc) * 0.8)
            front_num = int(len(pc) * 0.9)
        else:
            start_index = 0

        pc_front_mean = compute_pc_front_mean(pc, front_num, start_index)
        front_avg_dict[pc_name] = pc_front_mean
    
    ## compute the scale size of each obj
    x_length_dict, y_length_dict, x_pos_dict, y_pos_dict, max_obj_name = get_xy_scale(
        pc_dict,
        size_ratio=size_ratio,
    )

    ## get the scale factor of each object according to x_length and y_length
    gpt_length_dict, obj_length_dict, max_obj_name, max_scale = scale_obj(mesh_dict, x_length_dict, y_length_dict, max_obj_name, size_ratio)

    ## put each object
    all_obj_v, scene_v_dict = put_obj_dup(args, mesh_dict, front_avg_dict, gpt_length_dict, obj_length_dict, max_obj_name, max_scale, x_pos_dict, y_pos_dict)

    ## save the transformed scene
    scale_trans_dict = {}
    scale_save_scene(scene_v_dict, all_obj_v, output_dir, case_name, scale_trans_dict)


    



    return 


def duplicate_obj(duplicate_info, obj_dict, obj_name, obj_v, obj_path):
    return 

def load_obj_info_texture(obj_path):
    obj = trimesh.load(obj_path, process = False)
    obj_v = obj.vertices
    obj_v_x = obj_v[:, 0].reshape(-1, 1)
    obj_v_y = -obj_v[:, 1].reshape(-1, 1)
    obj_v_z = -obj_v[:, 2].reshape(-1, 1)
    obj_v_new = np.concatenate((obj_v_x, obj_v_y, obj_v_z), axis= 1) 

    return obj_v_new, obj, obj_path  


def scale_save_scene(scene_dict, all_obj_v, final_obj_prefix, case_name, scale_trans_dict, target_length = 2):

    ## scale and translate all the objects into the scene with the bounding box 2 * 2 * 2 and center at the origin [0,0,0]
    translate = -np.mean(all_obj_v, axis = 0)
    scale_x = target_length / (np.max(all_obj_v[:,0]) - np.min(all_obj_v[:,0]))
    scale_y = target_length / (np.max(all_obj_v[:,1]) - np.min(all_obj_v[:,1]))
    scale_z = target_length / (np.max(all_obj_v[:,2]) - np.min(all_obj_v[:,2]))
    scale = min(scale_x, scale_y, scale_z)

    scale_trans_dict['scene_trans'] = (translate).tolist()
    scale_trans_dict['scene_scale'] = float(scale)

    if os.path.exists(f'{final_obj_prefix}/{case_name}'):
        rm_command = f'rm -r {final_obj_prefix}/{case_name}'
        os.system(rm_command)
    os.makedirs(f'{final_obj_prefix}/{case_name}', exist_ok=True)

    for obj_name, obj in scene_dict.items():
        obj_v = obj['v']

        ori_obj_path = obj['obj_path']

        ## load the original obj file 
        with open(ori_obj_path, 'r') as f:
            obj_data = f.readlines()

        obj_data = rewrite_obj_vertex_lines(obj_data, obj_v)

        ## save the new obj file
        new_obj_path = f'{final_obj_prefix}/{case_name}/{obj_name}.obj'
        with open(new_obj_path, 'w') as f:
            f.writelines(obj_data)
    
        ## mv the material file and the mtl file to the destination
        os.system(f'cp {ori_obj_path.split(".obj")[0]}.mtl {final_obj_prefix}/{case_name}/{obj_name}.mtl')
        os.system(f'cp {ori_obj_path.split(".obj")[0]}.png {final_obj_prefix}/{case_name}/{obj_name}.png')


def put_obj_dup(args, color_obj_dict, front_avg_dict, gpt_length_dict, obj_length_dict, max_obj_name, max_scale, x_pos_dict, y_pos_dict):

    update_obj_v_dict = {}
    all_obj_v = []

    ## get current max lenght of the max object in the scene 
    max_obj_length = obj_length_dict[max_obj_name]
    #print('max_obj_length:', max_obj_length)
    #print('max_scale:', max_scale)
    #exit(0)
    max_obj_cur_length = max_obj_length / max_scale
    absolute_categories = _absolute_size_categories(gpt_length_dict)
    
    
    for obj_name, obj in color_obj_dict.items():

        obj_v = obj['v']
        textured_obj = obj['textured_obj']

        ## scale the obj
        #print('obj_name:', obj_name)

        obj_ori_lenghth = obj_length_dict[obj_name]
        if get_object_category(obj_name) in absolute_categories:
            obj_scale = 1.0
        else:
            obj_cur_length = (gpt_length_dict[obj_name] / gpt_length_dict[max_obj_name]) * max_obj_cur_length
            obj_scale = obj_cur_length / obj_ori_lenghth
        obj_v = obj_v * obj_scale
        

        if obj_name in front_avg_dict.keys():
            ## move the obj to make its min z align with the front mean position
            obj_v[:,2] = obj_v[:,2] - np.min(obj_v[:,2]) + front_avg_dict[obj_name][2]
            
            ## move the obj to make its avg x align with x_pos
            obj_v[:,0] = obj_v[:,0] - (np.min(obj_v[:,0]) + np.max(obj_v[:,0]))/2 + x_pos_dict[obj_name]
            
            ## move the obj to make its avg y align with y_pos
            obj_v[:,1] = obj_v[:,1] - (np.min(obj_v[:,1]) + np.max(obj_v[:,1]))/2 + y_pos_dict[obj_name]
        else:
                        
            ## move the obj to make its min z align with the front mean position
            obj_v[:,2] = obj_v[:,2] - np.min(obj_v[:,2]) 
            
            ## move the obj to make its avg x align with x_pos
            obj_v[:,0] = obj_v[:,0] - (np.min(obj_v[:,0]) + np.max(obj_v[:,0]))/2 
            
            ## move the obj to make its avg y align with y_pos
            obj_v[:,1] = obj_v[:,1] - (np.min(obj_v[:,1]) + np.max(obj_v[:,1]))/2 


        all_obj_v.append(obj_v)

        update_obj_v_dict[obj_name] = {}
        update_obj_v_dict[obj_name]['v'] = obj_v
        update_obj_v_dict[obj_name]['textured_obj'] = textured_obj
        update_obj_v_dict[obj_name]['obj_path'] = obj['obj_path']

    all_obj_v = np.concatenate(all_obj_v, axis = 0)

    return all_obj_v, update_obj_v_dict
        


def put_obj(color_obj_dict, front_avg_dict, gpt_length_dict, obj_length_dict, max_obj_name, max_scale, x_pos_dict, y_pos_dict):

    update_obj_v_dict = {}
    all_obj_v = []

    ## get current max lenght of the max object in the scene 
    max_obj_length = obj_length_dict[max_obj_name]
    #print('max_obj_length:', max_obj_length)
    #print('max_scale:', max_scale)
    #exit(0)
    max_obj_cur_length = max_obj_length / max_scale
    absolute_categories = _absolute_size_categories(gpt_length_dict)
    
    
    for obj_name, obj in color_obj_dict.items():

        obj_v = obj['v']
        textured_obj = obj['textured_obj']

        ## scale the obj
        #print('obj_name:', obj_name)

        obj_ori_lenghth = obj_length_dict[obj_name]
        if get_object_category(obj_name) in absolute_categories:
            obj_scale = 1.0
        else:
            obj_cur_length = (gpt_length_dict[obj_name] / gpt_length_dict[max_obj_name]) * max_obj_cur_length
            obj_scale = obj_cur_length / obj_ori_lenghth
        obj_v = obj_v * obj_scale
        
        ## move the obj to make its min z align with the front mean position
        obj_v[:,2] = obj_v[:,2] - np.min(obj_v[:,2]) + front_avg_dict[obj_name][2]
        
        ## move the obj to make its avg x align with x_pos
        obj_v[:,0] = obj_v[:,0] - np.mean(obj_v[:,0]) + x_pos_dict[obj_name]
        
        ## move the obj to make its avg y align with y_pos
        obj_v[:,1] = obj_v[:,1] - np.mean(obj_v[:,1]) + y_pos_dict[obj_name]

        all_obj_v.append(obj_v)

        update_obj_v_dict[obj_name] = {}
        update_obj_v_dict[obj_name]['v'] = obj_v
        update_obj_v_dict[obj_name]['textured_obj'] = textured_obj
        update_obj_v_dict[obj_name]['obj_path'] = obj['obj_path']

    all_obj_v = np.concatenate(all_obj_v, axis = 0)

    return all_obj_v, update_obj_v_dict
        

def load_pc_data(pc_path):
    
    pc_data = np.load(pc_path)
    pc_data = pc_data['point_cloud']

    return pc_data


def load_seg_items(seg_folder, case_name):
    def parse_seg_item_name(seg_item_stem):
        prefix = f"{case_name}_"
        remainder = seg_item_stem[len(prefix):] if seg_item_stem.startswith(prefix) else seg_item_stem
        return remainder.strip('_')

    seg_items_dict = {}
    seg_prefix = seg_folder
    ## load all the segmented items
    for seg_item in os.listdir(seg_prefix):
        if not seg_item.endswith('.png') or seg_item.endswith('_segmentation.png') or seg_item.endswith('_ann.png'):
            continue
        #object_count_num = seg_item.replace(case_name, '').split('_')[1]
        seg_item_name = parse_seg_item_name(seg_item.split('.')[0])
        seg_item_path = f'{seg_prefix}/{seg_item}'
        seg_item_data = cv2.imread(seg_item_path)
        seg_items_dict[seg_item_name] = seg_item_data
    return seg_items_dict

def split_point_cloud(pc_data, seg_items_dict):

    point_cloud_dict = {}

    for item_name, item_data in seg_items_dict.items():
        original_pixel_count = np.sum(item_data > 0)
        target_pixel_count = original_pixel_count * 0.1

        if item_data.ndim == 3:
            base_mask = cv2.cvtColor(item_data, cv2.COLOR_BGR2GRAY)
        else:
            base_mask = item_data.copy()
        base_mask = np.where(base_mask > 0, 255, 0).astype(np.uint8)
        if base_mask.shape != pc_data.shape[:2]:
            base_mask = cv2.resize(
                base_mask,
                (pc_data.shape[1], pc_data.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        if np.sum(base_mask > 0) == 0:
            point_cloud_dict[item_name] = pc_data[base_mask > 0]
            continue

        distance_to_boundary = cv2.distanceTransform(base_mask, cv2.DIST_C, 3)
        radius = 2
        selected_pc_num = np.sum(distance_to_boundary > radius)

        if selected_pc_num > target_pixel_count:
            max_radius = max(base_mask.shape)
            lower_radius = radius
            upper_radius = radius
            upper_selected_pc_num = selected_pc_num

            while upper_selected_pc_num > target_pixel_count and upper_radius < max_radius:
                upper_radius = min(max_radius, max(upper_radius + 1, upper_radius * 2))
                upper_selected_pc_num = np.sum(distance_to_boundary > upper_radius)

            if upper_selected_pc_num > target_pixel_count:
                radius = upper_radius
            else:
                while lower_radius + 1 < upper_radius:
                    mid_radius = (lower_radius + upper_radius) // 2
                    mid_selected_pc_num = np.sum(distance_to_boundary > mid_radius)
                    if mid_selected_pc_num > target_pixel_count:
                        lower_radius = mid_radius
                    else:
                        upper_radius = mid_radius
                radius = upper_radius

        eroded_item_data = np.where(distance_to_boundary > radius, 255, 0).astype(np.uint8)

        ## get the mask to filter the point cloud
        item_mask = eroded_item_data[:, :] > 0
        item_pc = pc_data[item_mask]
        point_cloud_dict[item_name] = item_pc

    '''
    ## export the point cloud data
    for obj_name in point_cloud_dict.keys():
        pc = point_cloud_dict[obj_name]
        save_path = f'debug/largestudy/{obj_name}.obj'
        draw_colored_points_to_obj(save_path, pc)
    '''


    return point_cloud_dict

def compute_pc_front_mean(pc, front_num, start_index):

    ## sort the pc by the z value from small to big
    pc = pc[pc[:,2].argsort()]
    ## select the front_num points
    pc = pc[start_index:front_num]
    return np.mean(pc, axis=0)


def _max_positive_size(values):
    positive_values = []
    for value in values or []:
        if isinstance(value, (int, float)) and float(value) > 0:
            positive_values.append(float(value))
    if not positive_values:
        return None
    return max(positive_values)


def choose_scale_anchor(pc_dict, size_ratio=None):
    semantic_candidates = []
    fallback_candidates = []

    for pc_name, pc in pc_dict.items():
        if len(pc) == 0:
            continue

        x_min = np.min(pc[:, 0])
        x_max = np.max(pc[:, 0])
        y_min = np.min(pc[:, 1])
        y_max = np.max(pc[:, 1])
        z_min = np.min(pc[:, 2])
        z_max = np.max(pc[:, 2])

        x_length = float(x_max - x_min)
        y_length = float(y_max - y_min)
        z_length = float(z_max - z_min)
        xz_footprint = x_length * z_length
        max_extent = max(x_length, y_length, z_length)

        fallback_candidates.append((xz_footprint, max_extent, y_length, pc_name))

        if size_ratio is None:
            continue

        category_name = get_object_category(pc_name)
        semantic_size = _max_positive_size(size_ratio.get(category_name))
        if semantic_size is None:
            continue
        semantic_candidates.append((semantic_size, xz_footprint, max_extent, y_length, pc_name))

    if semantic_candidates:
        return max(semantic_candidates)[-1]
    if fallback_candidates:
        return max(fallback_candidates)[-1]
    return None


def get_xy_scale(pc_dict, size_ratio=None):

    x_length_dict = {}
    y_length_dict = {}

    x_pos_dict = {}
    y_pos_dict = {}

    for pc_name, pc in pc_dict.items():
        #print('pc_name:', pc_name)
        x_min = np.min(pc[:,0])
        x_max = np.max(pc[:,0])
        y_min = np.min(pc[:,1])
        y_max = np.max(pc[:,1])

        x_length = x_max - x_min
        y_length = y_max - y_min

        x_pos = (x_max + x_min) / 2
        y_pos = (y_max + y_min) / 2

        x_length_dict[pc_name] = x_length
        y_length_dict[pc_name] = y_length
        
        x_pos_dict[pc_name] = x_pos
        y_pos_dict[pc_name] = y_pos

    max_obj_name = choose_scale_anchor(pc_dict, size_ratio=size_ratio)
    if max_obj_name is None:
        raise ValueError("pc_dict must contain at least one non-empty point cloud")

    return x_length_dict, y_length_dict, x_pos_dict, y_pos_dict, max_obj_name




def scale_obj(obj_dict, x_length_dict, y_length_dict, max_obj_name, gpt_size_info):

    scale_dict = {}
    


    ## get the size of the max object 
    max_obj_v = obj_dict[max_obj_name]['v']
    max_obj_x_length = np.max(max_obj_v[:,0]) - np.min(max_obj_v[:,0])
    max_obj_y_length = np.max(max_obj_v[:,1]) - np.min(max_obj_v[:,1])
    
    max_pc_x_length = x_length_dict[max_obj_name]
    max_pc_y_length = y_length_dict[max_obj_name]
    
    max_x_scale = max_obj_x_length / max_pc_x_length
    max_y_scale = max_obj_y_length / max_pc_y_length
    max_scale = min(max_x_scale, max_y_scale)

    #print('max_x_scale:', max_x_scale)
    #print('max_obj_name:', max_obj_name)
    #exit(0)

    scale_dict[max_obj_name] = max_scale

    obj_length_dict = {}
    # ## collect the max length of th objects
    for obj_name, obj_info in obj_dict.items():
        obj_v = obj_info['v']
        obj_x_length = np.max(obj_v[:,0]) - np.min(obj_v[:,0])
        obj_y_length = np.max(obj_v[:,1]) - np.min(obj_v[:,1])
        obj_z_length = np.max(obj_v[:,2]) - np.min(obj_v[:,2])
        obj_max_length = max(obj_x_length, obj_y_length, obj_z_length)
        obj_length_dict[obj_name] = obj_max_length

    ## collect the max length from size info 
    gpt_length_dict = {}
    for obj_name in obj_dict.keys():
        obj_category = get_object_category(obj_name)
        size_list = gpt_size_info[obj_category]    
        max_gpt_size = max(size_list)
        #max_gpt_size = size_list[0] * size_list[1] * size_list[2]
        gpt_length_dict[obj_name] = max_gpt_size
    absolute_categories = _absolute_size_categories(gpt_size_info)
    if absolute_categories:
        gpt_length_dict['__absolute_size_categories__'] = sorted(absolute_categories)
    

    return gpt_length_dict, obj_length_dict, max_obj_name, max_scale    





'''
def scale_obj_previous(obj_dict, x_length_dict, y_length_dict, max_obj_name, size_ratio):

    scale_dict = {}

    for obj_name, obj_info in obj_dict.items():

        obj = obj_info['v']
        obj_x_length = np.max(obj[:,0]) - np.min(obj[:,0])
        obj_y_length = np.max(obj[:,1]) - np.min(obj[:,1])

        pc_x_length = x_length_dict[obj_name]
        pc_y_length = y_length_dict[obj_name]

        x_scale = obj_x_length / pc_x_length
        y_scale = obj_y_length / pc_y_length

        ## get min of x_scale and y_scale
        scale = min(x_scale, y_scale)

        scale_dict[obj_name] = scale

    return scale_dict
'''

def get_object_category(obj_name):
    ## get obj category 
    category = [alpha for alpha in obj_name if not alpha.isdigit()]
    category = ''.join(category)

    return category 

def get_object_ratio(ratio_info, obj_dict, max_obj_name):

    ## get the max length of the max object 
    max_obj_v = obj_dict[max_obj_name]['v']
    max_obj_x_length = np.max(max_obj_v[:,0]) - np.min(max_obj_v[:,0])
    max_obj_y_length = np.max(max_obj_v[:,1]) - np.min(max_obj_v[:,1])
    max_obj_z_length = np.max(max_obj_v[:,2]) - np.min(max_obj_v[:,2])
    max_obj_length = max(max_obj_x_length, max_obj_y_length, max_obj_z_length)


    #max_obj_max_length = 
    #for obj_name, obj_info in obj_dict.items():#

    #    max


def get_initial_layout(input_folder, input_depth_folder, input_seg_folder, \
                       case_name, output_dir, front_num, size_ratio):

    scale_trans_dict = {}

    ## load the point cloud data
    pc_path = f'{input_depth_folder}/{case_name}/{case_name}_point_cloud.npz'
    pc_data = load_pc_data(pc_path)
    
    ## load the segmented items
    seg_folder = f'{input_seg_folder}/{case_name}'
    seg_items_dict = load_seg_items(seg_folder, case_name)
    
    ## seg the point cloud
    pc_dict = split_point_cloud(pc_data, seg_items_dict)

    ## load the obj meshes
    mesh_dict = {}
    for obj_file in os.listdir(f'{input_folder}/{case_name}'):
        if obj_file.split('.')[-1] == 'obj':
            obj_name = obj_file.split('.')[0]
            obj_path = f'{input_folder}/{case_name}/{obj_file}'
            obj_v, textured_obj, obj_path = load_obj_info_texture(obj_path)
            mesh_dict[obj_name] = {
                'v': obj_v,
                'textured_obj': textured_obj,
                'obj_path': obj_path
            }
    

    ## compute the front mean position of the object
    front_avg_dict = {}
    for pc_name, pc in pc_dict.items():

        pc_front_mean = compute_pc_front_mean(pc, front_num, start_index=0)
        front_avg_dict[pc_name] = pc_front_mean
    

    ## compute the scale size of each obj
    x_length_dict, y_length_dict, x_pos_dict, y_pos_dict, max_obj_name = get_xy_scale(
        pc_dict,
        size_ratio=size_ratio,
    )

    ## get the scale factor of each object according to x_length and y_length
    gpt_length_dict, obj_length_dict, max_obj_name, max_scale = scale_obj(mesh_dict, x_length_dict, y_length_dict, max_obj_name, size_ratio)

    ## put each object
    all_obj_v, scene_v_dict = put_obj(mesh_dict, front_avg_dict, gpt_length_dict, obj_length_dict, max_obj_name, max_scale, x_pos_dict, y_pos_dict)

    ## save the transformed scene
    scale_save_scene(scene_v_dict, all_obj_v, output_dir, case_name, scale_trans_dict)
