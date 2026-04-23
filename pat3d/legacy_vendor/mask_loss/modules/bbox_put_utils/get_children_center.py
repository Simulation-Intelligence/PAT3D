import numpy as np 
import trimesh
import json 
from modules.bbox_put_utils.get_xz_proj import get_xz_projection_from_ver
from modules.bbox_put_utils.put_obj import push_objects_into_container, judge_layer_intersection, solve_intersection_in_sub_layer

def load_scene_tree(file_path):
    with open(file_path, 'r') as file:
        scene_tree = json.load(file)
    return scene_tree


def get_node_bbox(node_name, optimized_node_information):
    if node_name in optimized_node_information:
        return optimized_node_information[node_name]
    else:
        raise ValueError(f"Node {node_name} not found in optimized node information.")


def from_center_to_bbox_bounds(son_node_center, son_node_name, geometry_prefix):
    ## load the geometry of the son node
    object_vertices = trimesh.load(f'{geometry_prefix}/{son_node_name}.obj', process=False).vertices
    object_x_length = np.max(object_vertices[:, 0]) - np.min(object_vertices[:, 0])
    object_y_length = np.max(object_vertices[:, 1]) - np.min(object_vertices[:, 1])
    object_z_length = np.max(object_vertices[:, 2]) - np.min(object_vertices[:, 2])

    object_bbox = {
        'x_min': son_node_center[0] - object_x_length / 2,
        'x_max': son_node_center[0] + object_x_length / 2,
        'y_min': son_node_center[1] - object_y_length / 2,
        'y_max': son_node_center[1] + object_y_length / 2,
        'z_min': son_node_center[2] - object_z_length / 2,
        'z_max': son_node_center[2] + object_z_length / 2
    }

    return object_bbox


## geometry_prefix: the folder prefix for the geometry files
## mom_node_bbox: the bounding box of the parent node {"xmin": xx, "xmax": xx, "ymax": xx, "zmin": xx, "zmax": xx} 
## mom_node_name: the name of the parent node
def get_children_center(mom_node_bbox, mom_node_name, geometry_prefix, scene_tree, gap_y):

    #print('mom node!!!:', mom_node_name)
    #print('mom node bbox:', mom_node_bbox)
    ## load the geometry of both the son and the mom objects 
    object_vertices_dict = load_geometry([mom_node_name] + scene_tree[mom_node_name], geometry_prefix)

    ## get the offset for the parent node
    x_offset, y_offset, z_offset = get_mom_node_offset(mom_node_bbox, object_vertices_dict[mom_node_name]) 
   
    #print('mom name', mom_node_name, 'x_offset', x_offset, 'y_offset', y_offset, 'z_offset', z_offset)

    ## get the new xz projection of the parent node with the offset 
    mom_xz_projection_contour, mom_bounds_dict = get_xz_projection_contour(mom_node_name, geometry_prefix, x_offset, y_offset, z_offset)

    #print('mom bounds:', mom_bounds_dict)

    ## move the son objects to the center of the parent node
    mom_center = match_son_cluster_and_mom_center(scene_tree[mom_node_name], object_vertices_dict, mom_node_name, mom_bounds_dict, mom_bounds_dict['y_max'] + gap_y)

    ## drag the objects into the container and update the vertices 
    ## will change the vertices of the son objects in object_vertices_dict
    push_objects_into_container(scene_tree[mom_node_name], object_vertices_dict, mom_xz_projection_contour, mom_bounds_dict)
    

    ## ===============================================================================
    ## start to solve the intersection problem within one layer 
    ## ===============================================================================

    ## judge if the layer has intersection 
    sublayer_object_order_list = scene_tree[mom_node_name].copy()

    #print('sublayer object order list:', sublayer_object_order_list)

    layer_starting_y = mom_bounds_dict['y_max'] + 0.005
    #print('layer starting y:', layer_starting_y)

    while len(sublayer_object_order_list) > 0:

        #print('==========================')
        print('sublayer object order list:', sublayer_object_order_list)

        intersection_flag = judge_layer_intersection(sublayer_object_order_list, object_vertices_dict)

        print('intersection flag:', intersection_flag)
        #input()

        if intersection_flag:
            print('Detect intersection in the layer, pushing out the objects')
            layer_starting_y, sublayer_object_order_list = solve_intersection_in_sub_layer(layer_starting_y, sublayer_object_order_list, object_vertices_dict, mom_center, mom_xz_projection_contour, mom_bounds_dict)
            
            ## update the y value of each object in the sub layer with larer_starting_y
            for obj_name in sublayer_object_order_list:
                obj_vertices = object_vertices_dict[obj_name]
                obj_vertices[:, 1] = layer_starting_y + obj_vertices[:, 1] - obj_vertices[:, 1].min()
                object_vertices_dict[obj_name] = obj_vertices
        else:
            break
    
    ## get the center of the son objects 
    son_node_center_dict = {}
    for son_node_name in scene_tree[mom_node_name]:
        son_object_vertices = object_vertices_dict[son_node_name]
        son_object_center = np.array([
            (np.max(son_object_vertices[:, 0]) + np.min(son_object_vertices[:, 0])) / 2,
            (np.max(son_object_vertices[:, 1]) + np.min(son_object_vertices[:, 1])) / 2,
            (np.max(son_object_vertices[:, 2]) + np.min(son_object_vertices[:, 2])) / 2 
        ])
        son_node_center_dict[son_node_name] = son_object_center

    #print('son_node_center_dict:', son_node_center_dict)

    #input()

    return  son_node_center_dict

## load the geometry vertices for each object from the original mesh folder
def load_geometry(object_name_list, geometry_prefix):
    
    #print(f"load geometry for {object_name_list}")
    object_vertices_dict = {}
    ## process the ground node 
    for object_name in object_name_list:
        if object_name == 'ground':
            ## just to build up the bounding box range for the ground
            object_vertices = np.array([
                    [-100, -1, -100],  
                    [100, -1, -100],   
                    [-100, -1, 100],   
                    [100, -1, 100],    
                    [0, -1, 0]])
            
        else:
            object_vertices = trimesh.load(f'{geometry_prefix}/{object_name}.obj', process=False).vertices
            object_vertices[:, 0] = object_vertices[:, 0] * -1
            object_vertices[:, 1] = object_vertices[:, 1] * -1
            
        object_vertices_dict[object_name] = object_vertices  

    return object_vertices_dict


def load_trimesh_item(object_name_list, geometry_prefix):
    
    #print(f"load geometry for {object_name_list}")
    object_vertices_dict = {}
    ## process the ground node 
    for object_name in object_name_list:
        
        object_item = trimesh.load(f'{geometry_prefix}/{object_name}.obj', process=False)
        object_item.vertices[:, 0] = object_item.vertices[:, 0] * (-1)
        object_item.vertices[:, 1] = object_item.vertices[:, 1] * (-1)
        object_vertices_dict[object_name] = object_item  

    return object_vertices_dict


def get_mom_node_offset(mom_node_bbox, mom_node_vertices):

    ## get the original xmin, xmax, ymin, ymax, zmin, zmax of the parent node from mom_node_vertices 
    ori_x_min = np.min(mom_node_vertices[:, 0])
    ori_x_max = np.max(mom_node_vertices[:, 0])
    ori_y_max = np.max(mom_node_vertices[:, 1])
    ori_z_min = np.min(mom_node_vertices[:, 2])
    ori_z_max = np.max(mom_node_vertices[:, 2])

    ## get the current xmin, xmax, ymin, ymax, zmin, zmax of the parent node from mom_node_bbox
    cur_x_min = mom_node_bbox['x_min']
    cur_x_max = mom_node_bbox['x_max']
    cur_y_max = mom_node_bbox['y_max']
    cur_z_min = mom_node_bbox['z_min']
    cur_z_max = mom_node_bbox['z_max']

    x_offset = cur_x_max - ori_x_max
    y_offset = cur_y_max - ori_y_max
    z_offset = cur_z_max - ori_z_max

    return x_offset, y_offset, z_offset

def get_xz_projection_contour(mom_node_name, geometry_prefix, x_offset, y_offset, z_offset):
    
    if mom_node_name == 'ground':

        mom_bounds_dict = {
            'x_min': -100,
            'x_max': 100,
            'y_max': -1,
            'z_min': -100,
            'z_max': 100
        }
        mom_xz_projection_points = np.array([
            [-100 ,-100],
            [ 100 ,-100],
            [ 100,  100],
            [-100 , 100]
        ])
    else:

        ## get the mom geometry vertices 
        mom_node_vertices = trimesh.load(f'{geometry_prefix}/{mom_node_name}.obj', process=False).vertices
        mom_node_vertices[:, 0] = mom_node_vertices[:, 0] * -1
        mom_node_vertices[:, 1] = mom_node_vertices[:, 1] * -1
        
        ## transformed the mom node vertices with the offset
        mom_node_vertices[:, 0] = mom_node_vertices[:, 0] + x_offset
        mom_node_vertices[:, 1] = mom_node_vertices[:, 1] + y_offset
        mom_node_vertices[:, 2] = mom_node_vertices[:, 2] + z_offset

        mom_x_min = np.min(mom_node_vertices[:, 0])
        mom_x_max = np.max(mom_node_vertices[:, 0])
        mom_y_max = np.max(mom_node_vertices[:, 1])
        mom_z_min = np.min(mom_node_vertices[:, 2])
        mom_z_max = np.max(mom_node_vertices[:, 2])
        mom_bounds_dict = {
            'x_min': mom_x_min,
            'x_max': mom_x_max,
            'y_max': mom_y_max,
            'z_min': mom_z_min,
            'z_max': mom_z_max
        }


        mom_xz_projection_points = get_xz_projection_from_ver(mom_node_vertices)

    return mom_xz_projection_points, mom_bounds_dict

def match_son_cluster_and_mom_center(cluster_x_order_list, mesh_obj_vertex_dict, mom_node_name, bounds_dict, layer_starting_y):

    ## compute the center of the cluster 
    son_cluster_vertices = []
    for obj_name in cluster_x_order_list:
        obj_vertices = mesh_obj_vertex_dict[obj_name]
        son_cluster_vertices.append(obj_vertices)
    son_cluster_vertices = np.concatenate(son_cluster_vertices, axis = 0)

    cluster_x_center = (son_cluster_vertices[:, 0].max() + son_cluster_vertices[:, 0].min()) / 2
    cluster_z_center = (son_cluster_vertices[:, 2].max() + son_cluster_vertices[:, 2].min()) / 2

    ## move all the cluster objects to the center of the container center 
    if not(mom_node_name == 'ground'):
        ## get the container center for those higher layer objects
        container_x_center = (bounds_dict['x_min'] + bounds_dict['x_max']) / 2
        container_z_center = (bounds_dict['z_min'] + bounds_dict['z_max']) / 2
        container_center = np.array([container_x_center, container_z_center])

        object_x_offset = container_x_center - cluster_x_center
        object_z_offset = container_z_center - cluster_z_center

    else:
        ## for ground layer, use the cluster center as the center of the container
        container_x_center = cluster_x_center 
        container_z_center = cluster_z_center
        container_center = np.array([container_x_center, container_z_center])

        object_x_offset = 0
        object_z_offset = 0

    ## update the cluster object vertices with the object_x_offset and object_z_offset
    ## update the cluster object vertices with the layer starting y
    for obj_name in cluster_x_order_list:
        obj_vertices = mesh_obj_vertex_dict[obj_name]
        obj_vertices[:, 0] += object_x_offset
        obj_vertices[:, 2] += object_z_offset
        obj_vertices[:, 1] = layer_starting_y + obj_vertices[:, 1] - obj_vertices[:, 1].min()
        mesh_obj_vertex_dict[obj_name] = obj_vertices

    return container_center