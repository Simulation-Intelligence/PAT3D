import os 
import trimesh 
import numpy as np
import matplotlib.pyplot as plt


def get_xz_projection(scene_name, low_poly_folder):
    
    ## load the object vertices 
    obj_prefix = os.path.join(low_poly_folder, scene_name)
    obj_path_list = os.listdir(obj_prefix)
    xz_projection_dict = {}
    xz_contour_dict = {}

    for obj_path in obj_path_list:

        if obj_path.endswith('_beforereduction.obj'):

            real_obj_path = os.path.join(obj_prefix, obj_path)
            obj_name = obj_path.split('.')[0]
            obj_vertices = trimesh.load(real_obj_path).vertices
            xz_vertices = get_ver_xz_projection(obj_vertices)

            ## get the x min and 
            x_min = xz_vertices[:, 0].min()
            x_max = xz_vertices[:, 0].max()

            #print(f'obj_name: {obj_name}, x_min: {x_min}, x_max: {x_max}')

            x_value_list = np.linspace(x_min, x_max, 50)
            
            ## for each x value, get the z bounds from xz vertices
            ## get the z min and max for each x value
            x_window_size = (x_max - x_min)/19
            for x_value in x_value_list:
                ## get the z value for each x value
                ## filter the points inside the window

                points_close = np.abs(xz_vertices[:, 0] - x_value) < x_window_size
                #print(f'obj_name: {obj_name}, x_value: {x_value}')
                z_values = xz_vertices[points_close, 1]
  
                z_min = z_values.min()
                z_max = z_values.max()
                if obj_name not in xz_projection_dict.keys():
                    xz_projection_dict[obj_name] = {}
                xz_projection_dict[obj_name][x_value] = (z_min, z_max)

            ## get all the boundary points 
            for x_value in x_value_list:
                contour_min_point = [x_value, xz_projection_dict[obj_name][x_value][0]]
                contour_max_point = [x_value, xz_projection_dict[obj_name][x_value][1]]
                if obj_name not in xz_contour_dict.keys():
                    xz_contour_dict[obj_name] = []
                xz_contour_dict[obj_name].append(contour_min_point)
                xz_contour_dict[obj_name].append(contour_max_point)
                
            xz_contour_dict[obj_name] = np.array(xz_contour_dict[obj_name])

    ## add ground range for the entire scene which is: x is from -100 to 100, z is from -100 to 100
    ground_xz_contour = np.array([[-100, -100], [100, -100], [100, 100], [-100, 100]])
    xz_contour_dict['ground'] = ground_xz_contour

    return xz_contour_dict



def get_low_xz_projection(scene_name, low_poly_folder):
    
    ## load the object vertices 
    obj_prefix = os.path.join(low_poly_folder, scene_name)
    obj_path_list = os.listdir(obj_prefix)
    xz_projection_dict = {}
    xz_contour_dict = {}

    for obj_path in obj_path_list:
        obj_name = obj_path.split('_')[0]

        if not (obj_path == f'{obj_name}.obj'):
            continue 
        else:
            real_obj_path = os.path.join(obj_prefix, obj_path)
            #print(f'real_obj_path: {real_obj_path}')
  
            obj_name = obj_path.split('.')[0]
            obj_vertices = trimesh.load(real_obj_path).vertices
            xz_vertices = get_ver_xz_projection(obj_vertices)

            ## get the x min and 
            x_min = xz_vertices[:, 0].min()
            x_max = xz_vertices[:, 0].max()

            #print(f'obj_name: {obj_name}, x_min: {x_min}, x_max: {x_max}')

            x_value_list = np.linspace(x_min, x_max, 50)
            
            ## for each x value, get the z bounds from xz vertices
            ## get the z min and max for each x value
            x_window_size = (x_max - x_min)/19
            for x_value in x_value_list:
                ## get the z value for each x value
                ## filter the points inside the window

                points_close = np.abs(xz_vertices[:, 0] - x_value) < x_window_size
                #print(f'obj_name: {obj_name}, x_value: {x_value}')
                z_values = xz_vertices[points_close, 1]
  
                z_min = z_values.min()
                z_max = z_values.max()
                if obj_name not in xz_projection_dict.keys():
                    xz_projection_dict[obj_name] = {}
                xz_projection_dict[obj_name][x_value] = (z_min, z_max)

            ## get all the boundary points 
            for x_value in x_value_list:
                contour_min_point = [x_value, xz_projection_dict[obj_name][x_value][0]]
                contour_max_point = [x_value, xz_projection_dict[obj_name][x_value][1]]
                if obj_name not in xz_contour_dict.keys():
                    xz_contour_dict[obj_name] = []
                xz_contour_dict[obj_name].append(contour_min_point)
                xz_contour_dict[obj_name].append(contour_max_point)
                
            xz_contour_dict[obj_name] = np.array(xz_contour_dict[obj_name])

    ## add ground range for the entire scene which is: x is from -100 to 100, z is from -100 to 100
    ground_xz_contour = np.array([[-100, -100], [100, -100], [100, 100], [-100, 100]])
    xz_contour_dict['ground'] = ground_xz_contour

    return xz_contour_dict



def get_ver_xz_projection(vertices):
    """
    获取形状在 x-z 平面的投影。
    
    Args:
        vertices: numpy array of shape (N, 3), 每行表示一个顶点 (x, y, z)
    
    Returns:
        xz_projection: numpy array of shape (N, 2)，每行存储投影点 (x, z)
    """
    # 提取 x 和 z 坐标
    xz_projection = vertices[:, [0, 2]]
    return xz_projection


def get_xz_projection_from_ver(obj_vertices):
     
    xz_vertices = get_ver_xz_projection(obj_vertices)

    ## get the x min and 
    x_min = xz_vertices[:, 0].min()
    x_max = xz_vertices[:, 0].max()

    x_value_list = np.linspace(x_min, x_max, 50)
    xz_projection_array = []
    xz_projection_dict = {}

    
    ## for each x value, get the z bounds from xz vertices
    ## get the z min and max for each x value
    x_window_size = (x_max - x_min)/19.0
    for x_value in x_value_list:
        ## get the z value for each x value
        ## filter the points inside the window

        points_close = np.abs(xz_vertices[:, 0] - x_value) < x_window_size
        z_values = xz_vertices[points_close, 1]
        z_min = z_values.min()
        z_max = z_values.max()
        xz_projection_dict[x_value] = (z_min, z_max)

    ## get all the boundary points 
    for x_value in x_value_list:
        contour_min_point = [x_value, xz_projection_dict[x_value][0]]
        contour_max_point = [x_value, xz_projection_dict[x_value][1]]
        xz_projection_array.append(contour_min_point)
        xz_projection_array.append(contour_max_point)
        
    xz_projection_array = np.array(xz_projection_array)

    return xz_projection_array