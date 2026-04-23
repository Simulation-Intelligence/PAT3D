import trimesh 
import numpy as np
import os

'''
def put_cluster( cluster_x_order_list, z_value_dict, x_center_value_dict, \
                 cluster_obj_vertex_dict, bounds_dict, xz_projection_dict, gap_y = 0.001, gap_x = 0.03):
    
    starting_left_front_bottom_point = [bounds_dict['x_min'], bounds_dict['y_min'], bounds_dict['z_min']]

    #print(f"Starting from: {starting_left_front_point}")
    
    cluster_position_dict = {}
    new_cluster_vertex_dict = {}
    cluster_largest_y = bounds_dict['y_min']
    for obj_name in cluster_x_order_list:

        obj_vertices = cluster_obj_vertex_dict[obj_name]
        obj_bbox_x_length = obj_vertices[:,0].max() - obj_vertices[:,0].min()
        obj_bbox_y_length = obj_vertices[:,1].max() - obj_vertices[:,1].min()
        obj_bbox_z_length = obj_vertices[:,2].max() - obj_vertices[:,2].min()

        #print(f"Object {obj_name} bbox length: {obj_bbox_x_length}, {obj_bbox_y_length}, {obj_bbox_z_length}")

        ## update the largest y for current layout 
        obj_y_max = starting_left_front_bottom_point[1] + obj_bbox_y_length 
        if obj_y_max > cluster_largest_y:
            cluster_largest_y = obj_y_max 

        ## judge if the object goes beyond the x bound 
        x_max = bounds_dict['x_max']
        #obj_x_end = starting_left_front_bottom_point[0] + obj_bbox_x_length/3 * 2
        obj_x_end = starting_left_front_bottom_point[0] + obj_bbox_x_length / 3 * 2

        
        ## if the object goes beyond the x bound, lift the layer to another layer
        ## change the starting point y to be the largest y of the previous layer
        ## change the starting point x to be the value that could make the object in the x bound
        if obj_x_end > x_max:
            print(f"Object {obj_name} goes beyond the x bound. Lift the layer to another layer.")
            starting_left_front_bottom_point[0] = x_max - obj_bbox_x_length
            starting_left_front_bottom_point[1] = cluster_largest_y + gap_y
            cluster_largest_y = starting_left_front_bottom_point[1] + obj_bbox_y_length

        
        ## get the center of the object  
        obj_center = [starting_left_front_bottom_point[0] + obj_bbox_x_length / 2, \
                      starting_left_front_bottom_point[1] + obj_bbox_y_length / 2, \
                      z_value_dict[obj_name] + obj_bbox_z_length / 2]

        ## clip the obj center if it goes beyond the z bound 
        obj_z_min = bounds_dict['z_min'] + obj_bbox_z_length / 2
        obj_z_max = bounds_dict['z_max'] - obj_bbox_z_length / 2
        obj_center[2] = np.clip(obj_center[2], obj_z_min, obj_z_max)

        ## update the center of the object
        cluster_position_dict[obj_name] = obj_center

        new_cluster_vertex_dict[obj_name] = update_new_cluster_position(obj_center, cluster_obj_vertex_dict[obj_name])

        ## update the starting point for the next object
        starting_left_front_bottom_point[0] += obj_bbox_x_length + gap_x

        
    
    return cluster_largest_y, cluster_position_dict, new_cluster_vertex_dict
'''


def put_cluster_new( cluster_x_order_list, z_front_value_dict, x_center_value_dict, 
                 cluster_obj_vertex_dict, bounds_dict, xz_projection_points, layer_num, gap_y = 0.02, gap_x = 0.008):
    
    cluster_position_dict = {}
    new_cluster_vertex_dict = {}
    cluster_objects_offset_dict = {}
    cluster_largest_y = bounds_dict['y_min']
    layer_starting_y = bounds_dict['y_min']

    for obj_name in cluster_x_order_list:

        print('processing object:', obj_name)

        obj_vertices = cluster_obj_vertex_dict[obj_name]
        obj_bbox_x_length = obj_vertices[:,0].max() - obj_vertices[:,0].min()
        obj_bbox_y_length = obj_vertices[:,1].max() - obj_vertices[:,1].min()
        obj_bbox_z_length = obj_vertices[:,2].max() - obj_vertices[:,2].min()

        ## make sure the object is in the x bound 
        x_center_range_min = bounds_dict['x_min'] + obj_bbox_x_length
        x_center_range_max = bounds_dict['x_max'] - obj_bbox_x_length 
    
        ## get the x center value of the object 
        obj_x_center = x_center_value_dict[obj_name]
        if x_center_range_min > x_center_range_max:
            obj_x_center1 = (x_center_range_min + x_center_range_max) / 2
            obj_x_center2 = x_center_range_max - obj_bbox_x_length / 2
            obj_x_center = max(obj_x_center1, obj_x_center2)
            #print('object x center wrong check')
            #print('center1:', obj_x_center1)
            #print('center2:', obj_x_center2)
            #print("obj bbox length:", obj_bbox_x_length)
        else:
            obj_x_center = np.clip(obj_x_center, x_center_range_min, x_center_range_max)

        ## get the z min and z max for the object 
        ## find the closest x value in the xz projection dict
        x_cloest_value = xz_projection_points[:, 0][np.abs(xz_projection_points[:, 0] - obj_x_center).argmin()]
        z_min_z_max_list = xz_projection_points[xz_projection_points[:, 0] == x_cloest_value]
        assert len(z_min_z_max_list) == 2, f"Error: the z min and max list is not 2 for {obj_name} with x value {x_cloest_value}"
        z_min = z_min_z_max_list[:, 1].min()
        z_max = z_min_z_max_list[:, 1].max()

        ## get the z value for the center of the object 
        z_front_value = z_front_value_dict[obj_name]

        if z_min + obj_bbox_z_length > z_max:
            obj_z_center = (z_min + z_max) / 2

        else:
            obj_z_center = np.clip(z_front_value + obj_bbox_z_length / 2, z_min + obj_bbox_z_length / 2, z_max - obj_bbox_z_length / 2)

        ## get the y value for the center of the object
        obj_y_center = layer_starting_y + obj_bbox_y_length / 2 

        ## update the largest y for current layout
        layer_starting_y = layer_starting_y + obj_bbox_y_length + gap_y

        ## update the center of the object
        obj_center = np.array([obj_x_center, obj_y_center, obj_z_center])
        cluster_position_dict[obj_name] = obj_center

        new_cluster_vertex_dict[obj_name], cur_object_offset = update_new_cluster_position(obj_center, cluster_obj_vertex_dict[obj_name])
        
        cluster_objects_offset_dict[obj_name] = cur_object_offset

    return new_cluster_vertex_dict, layer_starting_y, cluster_objects_offset_dict





def put_cluster_new_v2( cluster_x_order_list, mesh_obj_vertex_dict, bounds_dict, 
                 xz_projection_points, layer_num, gap_y = 0.01, gap_x = 0.008):

    cluster_largest_y = bounds_dict['y_min']
    layer_starting_y = bounds_dict['y_min']

    ## compute the center of the cluster 
    cluster_vertices = []
    for obj_name in cluster_x_order_list:
        obj_vertices = mesh_obj_vertex_dict[obj_name]
        cluster_vertices.append(obj_vertices)
    cluster_vertices = np.concatenate(cluster_vertices, axis = 0)

    cluster_x_center = (cluster_vertices[:, 0].max() + cluster_vertices[:, 0].min()) / 2
    cluster_z_center = (cluster_vertices[:, 2].max() + cluster_vertices[:, 2].min()) / 2

    ## move all the cluster objects to the center of the container center 
    if layer_num > 0:

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

        #if layer_num > 0:
        #layer_starting_y = layer_starting_y + obj_vertices[:, 1].max() - obj_vertices[:, 1].min() + gap_y

        ## update the layer highest y value 
        if cluster_largest_y < obj_vertices[:, 1].max():
            cluster_largest_y = obj_vertices[:, 1].max()
    
    ## drag the objects into the container and update the vertices 
    push_objects_into_container(cluster_x_order_list, mesh_obj_vertex_dict, xz_projection_points, bounds_dict)
    
    ## ===============================================================================
    ## start to solve the intersection problem within one layer 
    ## ===============================================================================

    ## judge if the layer has intersection 
    sublayer_object_order_list = cluster_x_order_list.copy()

    #print('sublayer object order list:', sublayer_object_order_list)

    while len(sublayer_object_order_list) > 0:
        #print('==========================')
        #print('layer starting y:', layer_starting_y)
        print('sublayer object order list:', sublayer_object_order_list)

        intersection_flag = judge_layer_intersection(sublayer_object_order_list, mesh_obj_vertex_dict)

        #print('intersection flag:', intersection_flag)
        #input()

        if intersection_flag:
            #print('Detect intersection in the layer, pushing out the objects')
            layer_starting_y, sublayer_object_order_list = solve_intersection_in_sub_layer(layer_starting_y, sublayer_object_order_list, mesh_obj_vertex_dict, container_center, xz_projection_points, bounds_dict)
            
            ## update the y value of each object in the sub layer with larer_starting_y
            for obj_name in sublayer_object_order_list:
                obj_vertices = mesh_obj_vertex_dict[obj_name]
                obj_vertices[:, 1] = layer_starting_y + obj_vertices[:, 1] - obj_vertices[:, 1].min()
                mesh_obj_vertex_dict[obj_name] = obj_vertices
                #print('object name:', obj_name)
                #print('y min:', obj_vertices[:,1].min())

        else:
            return mesh_obj_vertex_dict, cluster_largest_y
 
    return mesh_obj_vertex_dict, cluster_largest_y



def push_objects_into_container(cluster_x_order_list, sub_cluster_obj_vertex_dict, xz_projection_points, bounds_dict):

    x_min = bounds_dict['x_min']
    x_max = bounds_dict['x_max']

    for obj_name in cluster_x_order_list:

        obj_vertices = sub_cluster_obj_vertex_dict[obj_name]
        obj_bbox_x_length = obj_vertices[:,0].max() - obj_vertices[:,0].min()
        obj_bbox_z_length = obj_vertices[:,2].max() - obj_vertices[:,2].min()
        obj_current_x_center = (obj_vertices[:,0].max() + obj_vertices[:,0].min()) / 2
        obj_current_z_center = (obj_vertices[:,2].max() + obj_vertices[:,2].min()) / 2

        ## if the container is too small in x direction, just put the object in the center of the container
        if x_min + obj_bbox_x_length > x_max:
            obj_adjusted_x_center = (x_min + x_max) / 2
        ## else, clip the over flow region 
        else:
            x_center_range_min = x_min + obj_bbox_x_length/2
            x_center_range_max = x_max - obj_bbox_x_length/2
            obj_adjusted_x_center = np.clip(obj_current_x_center, x_center_range_min, x_center_range_max)
        
        ## move the object to make sure it is in the x bound
        obj_vertices[:,0] = obj_vertices[:,0] - obj_current_x_center + obj_adjusted_x_center

        ## get the z min and z max for the object
        ## find the closest x value in the xz projection dict
        x_cloest_value = xz_projection_points[:, 0][np.abs(xz_projection_points[:, 0] - obj_adjusted_x_center).argmin()]
        z_min_z_max_list = xz_projection_points[xz_projection_points[:, 0] == x_cloest_value]
        assert len(z_min_z_max_list) == 2, f"Error: the z min and max list is not 2 for {obj_name} with x value {x_cloest_value}"
        z_min = z_min_z_max_list[:, 1].min()
        z_max = z_min_z_max_list[:, 1].max()

        if z_min + obj_bbox_z_length > z_max:
            obj_adjusted_z_center = (z_min + z_max) / 2

        else:
            obj_adjusted_z_center = np.clip(obj_current_z_center, z_min + obj_bbox_z_length / 2, z_max - obj_bbox_z_length / 2)

        ## move the object to make sure it is in the z bound 
        obj_vertices[:,2] = obj_vertices[:,2] - obj_current_z_center + obj_adjusted_z_center

        ## update the vertices of the object
        sub_cluster_obj_vertex_dict[obj_name] = obj_vertices

    return sub_cluster_obj_vertex_dict


def put_cluster_uniform(args, cluster_x_order_list, mesh_obj_vertex_dict, bounds_dict, 
                 xz_projection_points, layer_num, mom_name, gap_y = 0.008, gap_x = 0.008):
    
    cluster_largest_y = bounds_dict['y_min']
    layer_starting_y = bounds_dict['y_min']

    ## get the x-z bounding box of the container
    container_x_min = bounds_dict['x_min']
    container_x_max = bounds_dict['x_max']
    container_z_min = bounds_dict['z_min']
    container_z_max = bounds_dict['z_max']

    object_random_offset_file_path = f'data/random_offset/{args.scene_name}_{layer_num}_{mom_name}.npz'
    
    if os.path.exists(object_random_offset_file_path):

        object_random_offset_dict = np.load(object_random_offset_file_path, allow_pickle=True)
        object_random_offset_dict = dict(object_random_offset_dict)
        print('object random offset dict:', object_random_offset_dict.keys())
        #print('cluster x order list:', cluster_x_order_list)

        for obj_name in cluster_x_order_list:

            ## get the object x length, y length and z length
            obj_vertices = mesh_obj_vertex_dict[obj_name]
            obj_bbox_x_length = obj_vertices[:,0].max() - obj_vertices[:,0].min()
            obj_bbox_y_length = obj_vertices[:,1].max() - obj_vertices[:,1].min()
            obj_bbox_z_length = obj_vertices[:,2].max() - obj_vertices[:,2].min()

            ## randomly generate a xz point in the container
            x_center_range_min = container_x_min + obj_bbox_x_length / 2
            x_center_range_max = container_x_max - obj_bbox_x_length / 2
            z_center_range_min = container_z_min + obj_bbox_z_length / 2
            z_center_range_max = container_z_max - obj_bbox_z_length / 2

            object_xz_center_point = object_random_offset_dict[obj_name] 

            object_x_center_value = object_xz_center_point[0][0]
            object_z_center_value = object_xz_center_point[0][1]
        
            obj_vertices = mesh_obj_vertex_dict[obj_name]
            ## move the object to the random xz point center 
            obj_vertices[:,0] = obj_vertices[:,0] - (obj_vertices[:,0].max() + obj_vertices[:,0].min()) / 2 + object_x_center_value
            obj_vertices[:,2] = obj_vertices[:,2] - (obj_vertices[:,2].max() + obj_vertices[:,2].min()) / 2 + object_z_center_value

            ## move object y to above the layer starting y
            obj_vertices[:,1] = layer_starting_y + obj_vertices[:,1] - obj_vertices[:,1].min()

            ## update the layer starting y value 
            layer_starting_y = layer_starting_y + obj_bbox_y_length + gap_y

                
            mesh_obj_vertex_dict[obj_name] = obj_vertices

        ## drag the objects into the container and update the vertices 
        push_objects_into_container(cluster_x_order_list, mesh_obj_vertex_dict, xz_projection_points, bounds_dict)

    else:
        object_random_offset_dict = {}

        for obj_name in cluster_x_order_list:

            ## get the object x length, y length and z length
            obj_vertices = mesh_obj_vertex_dict[obj_name]
            obj_bbox_x_length = obj_vertices[:,0].max() - obj_vertices[:,0].min()
            obj_bbox_y_length = obj_vertices[:,1].max() - obj_vertices[:,1].min()
            obj_bbox_z_length = obj_vertices[:,2].max() - obj_vertices[:,2].min()

            ## randomly generate a xz point in the container
            x_center_range_min = container_x_min + obj_bbox_x_length / 2
            x_center_range_max = container_x_max - obj_bbox_x_length / 2
            z_center_range_min = container_z_min + obj_bbox_z_length / 2
            z_center_range_max = container_z_max - obj_bbox_z_length / 2

            object_xz_center_point = np.random.rand(1, 2) * np.array([x_center_range_max - x_center_range_min, z_center_range_max - z_center_range_min]) + np.array([x_center_range_min, z_center_range_min])
            object_random_offset_dict[obj_name] = object_xz_center_point

            object_x_center_value = object_xz_center_point[0][0]
            object_z_center_value = object_xz_center_point[0][1]
        
            obj_vertices = mesh_obj_vertex_dict[obj_name]
            ## move the object to the random xz point center 
            obj_vertices[:,0] = obj_vertices[:,0] - (obj_vertices[:,0].max() + obj_vertices[:,0].min()) / 2 + object_x_center_value
            obj_vertices[:,2] = obj_vertices[:,2] - (obj_vertices[:,2].max() + obj_vertices[:,2].min()) / 2 + object_z_center_value

            ## move object y to above the layer starting y
            obj_vertices[:,1] = layer_starting_y + obj_vertices[:,1] - obj_vertices[:,1].min()

            ## update the layer starting y value 
            layer_starting_y = layer_starting_y + obj_bbox_y_length + gap_y

                
            mesh_obj_vertex_dict[obj_name] = obj_vertices

        ## drag the objects into the container and update the vertices 
        push_objects_into_container(cluster_x_order_list, mesh_obj_vertex_dict, xz_projection_points, bounds_dict)

        ## save the random offset dict to file
        with open(object_random_offset_file_path, 'wb') as f:
            np.savez(f, **object_random_offset_dict)

    #print('bounds_dict:', layer_starting_y)
    return mesh_obj_vertex_dict, cluster_largest_y
    



def solve_intersection_in_sub_layer(layer_starting_y, sublayer_object_order_list, mesh_obj_vertex_dict, \
                                     container_center, xz_projection_points, bounds_dict):
    
    #print('layer_starting_y:', layer_starting_y)
    #print('sublayer_object_order_list:', sublayer_object_order_list)
    #print('mesh_obj_vertex_dict:', mesh_obj_vertex_dict.keys())
    #print('container center:', container_center)
    #print('xz_projection_points:', xz_projection_points)
    #print('bounds_dict:', bounds_dict)
    #exit(0)
    ## get the container center for the current layer
    #import pdb 
    #pdb.set_trace()

    remaining_problem_object_order_list = []

    sublayer_object_offset_dict = {}

    ## step 1: get the center point of each object to find the closest object to the container center as the fixed object
    center_object_name = None 
    closest_distance_from_center = 10000000

    ## get the center point of each object in the scene 
    ## compute the distance between the object center and the container center
    for obj_name in sublayer_object_order_list:
        obj_vertices = mesh_obj_vertex_dict[obj_name]
        obj_x_center = (obj_vertices[:,0].max() + obj_vertices[:,0].min()) / 2
        obj_z_center = (obj_vertices[:,2].max() + obj_vertices[:,2].min()) / 2
        obj_container_distance = np.linalg.norm(np.array([obj_x_center, obj_z_center]) - container_center)
        if obj_container_distance < closest_distance_from_center:
            closest_distance_from_center = obj_container_distance
            center_object_name = obj_name
    sublayer_object_offset_dict[center_object_name] = np.array([0, 0, 0])

    #print('center object name:', center_object_name)

    #center_object_name = 'pillow1'

    ## step 2: get the x order of the objects from center to the boundary
    center_to_left_order_list = []
    center_to_right_order_list = []
    add_left_part = True
    for obj_name in sublayer_object_order_list:
        if add_left_part:
            if obj_name == center_object_name:
                add_left_part = False
                continue
            else:
                center_to_left_order_list.append(obj_name)
        else:
            center_to_right_order_list.append(obj_name)
        
    center_to_left_order_list.reverse()

    #print('center to left order list:', center_to_left_order_list) # center to left order list: ['orange1', 'apple1', 'banana1']
    #print('center to right order list:', center_to_right_order_list) # center to right order list: ['plum1']
    #print('center object name:', center_object_name) # center object name: pear1
    #exit(0)
    #input()

    ## step 3: remove the intersection from center to right 
    processed_object_name_list = []
    processed_object_name_list.append(center_object_name)
    for obj_name in center_to_right_order_list:
        
        newly_added_obj_vertices = mesh_obj_vertex_dict[obj_name]

        ## get the temporal vertices and mvoe the object 
        ## do not update the real mesh vertices until we finalize the position of the object
        newly_added_obj_vertices_temporary = newly_added_obj_vertices.copy()
        newly_added_obj_bbox = get_obj_bbox(newly_added_obj_vertices_temporary)

        intersection_check_pass_flag = False
        while intersection_check_pass_flag == False: 
            ## go over the processed object list to check if there is intersection
            one_pass_no_intersection_flag = True
            for processed_obj_name in processed_object_name_list:
                
                processed_obj_vertices = mesh_obj_vertex_dict[processed_obj_name]
                processed_obj_bbox = get_obj_bbox(processed_obj_vertices)
                intersection = compute_intersection(newly_added_obj_bbox, processed_obj_bbox)

                if intersection is not None:
                    #print('intersection detected between:', obj_name, 'and', processed_obj_name)
                    dx_positive, dx_negative, dz_positive, dz_negative = compute_separation_distances_for_B(processed_obj_bbox, newly_added_obj_bbox)
                    ## get the min distance to push the object out of the intersection and the according direction 
                    min_distance = min(dx_positive, dz_positive, dz_negative)
                    if min_distance == dx_positive:
                        #print('push the object to the right')
                        newly_added_obj_vertices_temporary[:,0] += min_distance
                    elif min_distance == dz_positive:
                        #print('push the object to the bottom')
                        newly_added_obj_vertices_temporary[:,2] += min_distance
                    elif min_distance == dz_negative:  
                        #print('push the object to the top')
                        newly_added_obj_vertices_temporary[:,2] -= min_distance

                    one_pass_no_intersection_flag = False 

                    newly_added_obj_bbox = get_obj_bbox(newly_added_obj_vertices_temporary)

            if one_pass_no_intersection_flag == False:
                intersection_check_pass_flag = False
            else:
                intersection_check_pass_flag = True
        
        ## if finish moving the object, update the vertices and the offset
        obj_offset_x = newly_added_obj_vertices_temporary[:, 0].min() - newly_added_obj_vertices[:, 0].min()
        obj_offset_z = newly_added_obj_vertices_temporary[:, 2].min() - newly_added_obj_vertices[:, 2].min()
        sublayer_object_offset_dict[obj_name] = np.array([obj_offset_x, 0, obj_offset_z])
        ## finalize the position of the object
        mesh_obj_vertex_dict[obj_name] = newly_added_obj_vertices_temporary

        processed_object_name_list.append(obj_name)


    ## step 4: remove the intersection from center to left
    #processed_object_name_list = []
    processed_object_name_list.append(center_object_name)
    for obj_name in center_to_left_order_list:

        #print('processing object:', obj_name)
        
        newly_added_obj_vertices = mesh_obj_vertex_dict[obj_name]

        ## get the temporal vertices and mvoe the object 
        ## do not update the real mesh vertices until we finalize the position of the object
        newly_added_obj_vertices_temporary = newly_added_obj_vertices.copy()
        newly_added_obj_bbox = get_obj_bbox(newly_added_obj_vertices_temporary)

        intersection_check_pass_flag = False
        while intersection_check_pass_flag == False: 
            ## go over the processed object list to check if there is intersection
            one_pass_no_intersection_flag = True
            for processed_obj_name in processed_object_name_list:

                #print('processed object name:', processed_obj_name)

                processed_obj_vertices = mesh_obj_vertex_dict[processed_obj_name]
                processed_obj_bbox = get_obj_bbox(processed_obj_vertices)
                intersection = compute_intersection(newly_added_obj_bbox, processed_obj_bbox)
                if intersection is not None:
                    #print('intersection detected between:', obj_name, 'and', processed_obj_name)
                    #input()
                    dx_positive, dx_negative, dz_positive, dz_negative = compute_separation_distances_for_B(processed_obj_bbox, newly_added_obj_bbox)
                    ## get the min distance to push the object out of the intersection and the according direction 
                    min_distance = min(dx_negative, dz_positive, dz_negative)
                    if min_distance == dx_negative:
                        #print('push the object to the right')
                        newly_added_obj_vertices_temporary[:,0] -= min_distance
                    elif min_distance == dz_positive:
                        #print('push the object to the bottom')
                        newly_added_obj_vertices_temporary[:,2] += min_distance
                    elif min_distance == dz_negative:  
                        #print('push the object to the top')
                        newly_added_obj_vertices_temporary[:,2] -= min_distance

                    one_pass_no_intersection_flag = False 
                    newly_added_obj_bbox = get_obj_bbox(newly_added_obj_vertices_temporary)

            if one_pass_no_intersection_flag == False:
                intersection_check_pass_flag = False
            else:
                intersection_check_pass_flag = True
        
        ## if finish moving the object, update the vertices and the offset
        obj_offset_x = newly_added_obj_vertices_temporary[:, 0].min() - newly_added_obj_vertices[:, 0].min()
        obj_offset_z = newly_added_obj_vertices_temporary[:, 2].min() - newly_added_obj_vertices[:, 2].min()
        #print('obj name:', obj_name)
        #print('obj offset x:', obj_offset_x)
        #print('obj offset z:', obj_offset_z)
        #input()
        sublayer_object_offset_dict[obj_name] = np.array([obj_offset_x, 0, obj_offset_z])
        
        ## finalize the position of the object
        mesh_obj_vertex_dict[obj_name] = newly_added_obj_vertices_temporary

        processed_object_name_list.append(obj_name)


    ## step 5: get the objects with problem, and move the layer up to another layer 
    ## roll back the mesh vertices of the problem objects
    ## if the object is not in the container's xz bound, roll back the object's vertices and put it into the remaining problem list
    sublayer_object_order_list_no_center_obj = sublayer_object_order_list.copy()
    sublayer_object_order_list_no_center_obj.remove(center_object_name)
    for obj_name in sublayer_object_order_list_no_center_obj:
        obj_x_center = (mesh_obj_vertex_dict[obj_name][:,0].max() + mesh_obj_vertex_dict[obj_name][:,0].min()) / 2
        obj_z_center = (mesh_obj_vertex_dict[obj_name][:,2].max() + mesh_obj_vertex_dict[obj_name][:,2].min()) / 2
        obj_x_length = mesh_obj_vertex_dict[obj_name][:,0].max() - mesh_obj_vertex_dict[obj_name][:,0].min()
        obj_z_length = mesh_obj_vertex_dict[obj_name][:,2].max() - mesh_obj_vertex_dict[obj_name][:,2].min()

        ## if the object is not in the container's x bound, roll back the object's vertices and put it into the remaining problem list
        if obj_x_center < bounds_dict['x_min'] + obj_x_length / 2 or obj_x_center > bounds_dict['x_max'] - obj_x_length / 2:
            
            remaining_problem_object_order_list.append(obj_name)
            ## roll back the vertices
            mesh_obj_vertex_dict[obj_name][:,0] -= sublayer_object_offset_dict[obj_name][0]
            mesh_obj_vertex_dict[obj_name][:,2] -= sublayer_object_offset_dict[obj_name][2]
            sublayer_object_offset_dict[obj_name] = np.array([0, 0, 0])

        else:

            ## get the z min and z max for the object 
            ## find the closest x value in the xz projection dict
            x_cloest_value = xz_projection_points[:, 0][np.abs(xz_projection_points[:, 0] - obj_x_center).argmin()]
            z_min_z_max_list = xz_projection_points[xz_projection_points[:, 0] == x_cloest_value]
            assert len(z_min_z_max_list) == 2, f"Error: the z min and max list is not 2 for {obj_name} with x value {x_cloest_value}"
            z_min = z_min_z_max_list[:, 1].min()
            z_max = z_min_z_max_list[:, 1].max()

            z_center_min = z_min + obj_z_length / 2
            z_center_max = z_max - obj_z_length / 2
            if obj_z_center < z_center_min or obj_z_center > z_center_max:
                remaining_problem_object_order_list.append(obj_name)
                ## roll back the vertices
                mesh_obj_vertex_dict[obj_name][:,0] -= sublayer_object_offset_dict[obj_name][0]
                mesh_obj_vertex_dict[obj_name][:,2] -= sublayer_object_offset_dict[obj_name][2]
                sublayer_object_offset_dict[obj_name] = np.array([0, 0, 0])

    
    #print('remaining problem object order list:', remaining_problem_object_order_list)
    #exit(0)
    #print('sublayer_object_order_list:', sublayer_object_order_list)

    ## step 6: update the highest y value of current sub layer 
    for obj_name in sublayer_object_order_list:
        if obj_name in processed_object_name_list:
            obj_y_max = mesh_obj_vertex_dict[obj_name][:,1].max()
            #print(obj_name, '|| y max:', obj_y_max)
            if obj_y_max > layer_starting_y:
                layer_starting_y = obj_y_max + 0.005
        

    return layer_starting_y, remaining_problem_object_order_list

                    


        













def judge_layer_intersection(cluster_x_order_list, cluster_obj_vertex_dict):
    
    for obj_name_a in cluster_x_order_list:
        for obj_name_b in cluster_x_order_list:
            if obj_name_a == obj_name_b:
                continue
            
            obj_a_vertices = cluster_obj_vertex_dict[obj_name_a]
            obj_b_vertices = cluster_obj_vertex_dict[obj_name_b]
            obj_a_bbox = get_obj_bbox(obj_a_vertices)
            obj_b_bbox = get_obj_bbox(obj_b_vertices)
            intersection = compute_intersection(obj_a_bbox, obj_b_bbox)
            
            if intersection is not None:
                return True 
            
    return False


def get_obj_bbox(obj_vertices):

    obj_bbox_left = obj_vertices[:,0].min()
    obj_bbox_top = obj_vertices[:,2].min()
    obj_bbox_right = obj_vertices[:,0].max()
    obj_bbox_bottom = obj_vertices[:,2].max()
    obj_bbox = (obj_bbox_left, obj_bbox_top, obj_bbox_right, obj_bbox_bottom)

    return obj_bbox




def compute_intersection(boxA, boxB):
    # Box A: (A_left, A_top, A_right, A_bottom)
    # Box B: (B_left, B_top, B_right, B_bottom)

    # Calculate intersection coordinates
    I_left = max(boxA[0], boxB[0])
    I_top = max(boxA[1], boxB[1])
    I_right = min(boxA[2], boxB[2])
    I_bottom = min(boxA[3], boxB[3])

    # Check if there is an intersection
    if I_right > I_left and I_bottom > I_top:
        # Return intersection coordinates
        return (I_left, I_top, I_right, I_bottom)
    else:
        # No intersection
        return None


def filter_order(x_order, cluster_obj_name_list):
    ## filter the x order to only keep the objects in the cluster
    filtered_x_order = []
    for obj_name in x_order:
        if obj_name in cluster_obj_name_list:
            filtered_x_order.append(obj_name)
    return filtered_x_order

def update_new_cluster_position(obj_center, obj_vertices):

    ori_x_center = (obj_vertices[:,0].max() + obj_vertices[:,0].min())/2
    ori_y_center = (obj_vertices[:,1].max() + obj_vertices[:,1].min())/2
    ori_z_center = (obj_vertices[:,2].max() + obj_vertices[:,2].min())/2
    ori_center = np.array([ori_x_center, ori_y_center, ori_z_center])
    obj_vertices = obj_vertices - ori_center
    obj_vertices = obj_vertices + obj_center

    ## get the offset of the current object 
    current_offset = obj_center - ori_center

    return obj_vertices, current_offset
        

def pick_object_name(category_name, x_order):
    for obj_name in x_order:
        current_category_name = get_category_name(obj_name)
        if current_category_name == category_name:
            x_order.remove(obj_name)
            return obj_name
    return 



def get_bottom_object_name(category_name, remaining_y_order):
    ## get the bottom object name in the y order
    bottom_obj_name = None
    for obj_name in remaining_y_order:
        current_category_name = get_category_name(obj_name)
        if current_category_name == category_name:
            bottom_obj_name = obj_name
            break
    return bottom_obj_name
    


def get_category_name(object_name):
    ## remove the digit from the object name
    category_name = ''.join(filter(lambda x: not x.isdigit(), object_name))
    return category_name


def compute_separation_distances_for_B(boxA, boxB):

    # 分别计算四个方向的最小移动距离
    dx_positive = boxA[2] - boxB[0]  # B 向右移动
    dx_negative = boxB[2] - boxA[0]  # B 向左移动
    dz_positive = boxA[3] - boxB[1]  # B 向下移动
    dz_negative = boxB[3] - boxA[1]  # B 向上移动

    # 如果某个方向已经无交集，则移动距离为 0
    dx_positive = max(0, dx_positive)
    dx_negative = max(0, dx_negative)
    dz_positive = max(0, dz_positive)
    dz_negative = max(0, dz_negative)

    return dx_positive, dx_negative, dz_positive, dz_negative


if __name__  == "__main__":
    # 示例
    boxA = (0, 1, 4, 3)  # A 的左上角 (1,1)，右下角 (3,3)
    boxB = (2, 2, 4, 6)  # B 的左上角 (2,2)，右下角 (4,4)

    # 计算将 B 移动到无交集的最小距离
    separation_distances = compute_separation_distances_for_B(boxA, boxB)
    print("B 的四个方向的移动距离：", separation_distances)

    ## 画出box A 和box B的2d可视化
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    fig, ax = plt.subplots()
    ax.add_patch(patches.Rectangle((boxA[0], boxA[1]), boxA[2]-boxA[0], boxA[3]-boxA[1], edgecolor='blue', facecolor='none', lw=2))
    ax.add_patch(patches.Rectangle((boxB[0], boxB[1]), boxB[2]-boxB[0], boxB[3]-boxB[1], edgecolor='red', facecolor='none', lw=2))
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 7)
    ax.set_aspect('equal', adjustable='box')
    plt.grid()
    plt.show()
    