import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))  
sys.path.append(project_root)

import numpy as np
import trimesh
import argparse



def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case_name', type=str, default=None)
    parser.add_argument('--input_root', type=str, default='data/layout')
    parser.add_argument('--input_depth_root', type=str, default='data/depth')
    parser.add_argument('--output_root', type=str, default='data/layout')
    return parser.parse_args()


def load_geometry(input_geo_prefix, geometry_case_name, save_geo_prefix):

    geometry_folder_prefix = f'{input_geo_prefix}/{geometry_case_name}/'
    save_geometry_folder_prefix = f'{save_geo_prefix}/{geometry_case_name}'
    ## list all the obj files in the folder
    geometry_files = os.listdir(geometry_folder_prefix)

    obj_dict = {}
    for obj_file in geometry_files:
        if obj_file[-3:] == 'obj':
            obj_name = obj_file.replace('.obj', '')
            obj_dict[obj_name] = {}

            ## get the vertices 
            obj_mesh = trimesh.load_mesh(f'{geometry_folder_prefix}{obj_file}')
            vertices = obj_mesh.vertices
            obj_dict[obj_name]['vertices'] = vertices

            ## load the obj file content 
            obj_dict[obj_name]['obj_file'] = load_obj_file(f'{geometry_folder_prefix}{obj_file}')

            ## make the save prefix 
            obj_dict[obj_name]['input_prefix'] = f'{geometry_folder_prefix}'
            obj_dict[obj_name]['save_prefix'] = f'{save_geometry_folder_prefix}'

    return obj_dict


def load_obj_file(obj_file_path):
    with open(obj_file_path, 'r') as file:  
        obj_file_info = file.readlines()    
    return obj_file_info


def get_ground_from_point_cloud(input_depth_prefix, geometry_case_name):

    ## load the point cloud
    depth_file_path = f'{input_depth_prefix}/{geometry_case_name}/{geometry_case_name}_point_cloud.ply'
    point_cloud = trimesh.load(depth_file_path)

    ## get the vertices
    vertices = point_cloud.vertices

    ## get the y value of the vertices
    y_values = vertices[:, 1]

    ## get the ground y value
    ground_y_value = np.max(y_values)

    return ground_y_value


def save_ground_y_value(output_root, case_name, ground_y_value):

    save_file_path = f'{output_root}/{case_name}/ground_y_value.txt'
    with open(save_file_path, 'w') as file:
        file.write(str(ground_y_value))
    file.close()


def move_object_above_floor(obj_dict, ground_y_value, ):

    for obj_name in obj_dict.keys():
        vertices = obj_dict[obj_name]['vertices']
        max_v_value = np.max(vertices[:, 1])
        if max_v_value > ground_y_value:
            move_value = ground_y_value - max_v_value

            ## move the object to be above the ground
            vertices[:, 1] += move_value

            ## save the new vertices
            obj_dict[obj_name]['vertices'] = vertices
    


    for obj_name in obj_dict.keys():

        vertices = obj_dict[obj_name]['vertices']
        
        ## export the new files 
        obj_file_info = obj_dict[obj_name]['obj_file']
        save_folder_prefix = obj_dict[obj_name]['save_prefix']
        input_folder_prefix = obj_dict[obj_name]['input_prefix']

        export_self(obj_file_info, vertices, save_folder_prefix, obj_name, input_folder_prefix)


def export_self(obj_file_info, translated_vertices, object_folder_prefix, obj_name, geometry_folder_prefix):

    ## mkdir the folder if not exist
    os.makedirs(object_folder_prefix, exist_ok=True)


    ## replace the vertices in the obj file
    ## the target information format v 0.32523788 -0.96331519 0.41444887
    formatted_lines = [f"v {xyz[0]:.8f} {xyz[1]:.8f} {xyz[2]:.8f}\n" for xyz in translated_vertices]
    
    ## replace the vertices in the obj file

    total_vertices_num = len(translated_vertices)
    obj_file_info[3:total_vertices_num + 3] = formatted_lines
    ## save the obj file
    save_obj_file_path = f'{object_folder_prefix}/{obj_name}.obj'
    with open(save_obj_file_path, 'w') as file:
        file.writelines(obj_file_info)

    '''
    ## only useful when the source folder and the target folder are different
    ## copy the texture png and mtl to the target folder 
    mtl_path = f'{geometry_folder_prefix}/{obj_name}.mtl'
    png_path = f'{geometry_folder_prefix}/{obj_name}.png'
    save_mtl_path = f'{object_folder_prefix}/{obj_name}.mtl'
    save_png_path = f'{object_folder_prefix}/{obj_name}.png'
    os.system(f'cp {mtl_path} {save_mtl_path}')
    os.system(f'cp {png_path} {save_png_path}')
    '''


def reset_ground(input_geometry_root, input_depth_root, scene_name, output_geometry_root):
    
    ## load the geometry
    obj_dict = load_geometry(input_geometry_root, scene_name, output_geometry_root)

    ## get the ground y value from the point cloud 
    ground_y_value = get_ground_from_point_cloud(input_depth_root, scene_name)

    ## compute the lowest point of the scene in the y axis (the biggest y value)
    ## move the object to be higher than the ground y value
    move_object_above_floor(obj_dict, ground_y_value)

    ## save the y ground value to the layout folder
    save_ground_y_value(output_geometry_root, scene_name, ground_y_value)




if __name__ == "__main__":
    args = get_config()

    ## load the geometry
    obj_dict = load_geometry(args.input_root, args.case_name, args.output_root)

    ## get the ground y value from the point cloud 
    ground_y_value = get_ground_from_point_cloud(args.input_depth_root, args.case_name)

    ## compute the lowest point of the scene in the y axis (the biggest y value)
    ## move the object to be higher than the ground y value
    move_object_above_floor(obj_dict, ground_y_value)

    ## save the y ground value to the layout folder
    save_ground_y_value(args.output_root, args.case_name, ground_y_value)








