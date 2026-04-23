import argparse
import pprint


def parse_options(return_parser=False, argv=None):
    # New CLI parser
    parser = argparse.ArgumentParser(description='PAT3D: Physics-Augmented Text-to-3D Scene Generation')
    
    # Global arguments
    global_group = parser.add_argument_group('global')
    global_group.add_argument('--exp_name', type = str,
                              help='Experiment name.')
    global_group.add_argument('--scene_name', type = str,
                              help='The name for the scene.')
    global_group.add_argument('--preprocess', action='store_true',
                              help='Preprocess to get the initial scene.')
    global_group.add_argument('--layout_init', action='store_true',
                              help='Layout initialization.')
    global_group.add_argument('--phys_optim', action='store_true',
                              help='Use physical optimization.')
    global_group.add_argument('--visualize', action='store_true',
                              help='Visualize the results after optimization.')
    global_group.add_argument('--runtime-config', type=str,
                              help='Path to runtime config JSON for canonical pipeline execution.')
    global_group.add_argument('--request', type=str,
                              help='Path to canonical SceneRequest JSON for runtime execution.')
    global_group.add_argument('--object-catalog', type=str,
                              help='Optional path to canonical ObjectCatalog JSON for runtime execution.')
    global_group.add_argument('--object-hints', type=str,
                              help='Optional path to JSON list of object hints for runtime execution.')
    global_group.add_argument('--object-reference-images', type=str,
                              help='Optional path to JSON mapping of object ids to ArtifactRef payloads.')
    global_group.add_argument('--output', type=str,
                              help='Optional path to write canonical runtime output JSON.')
    global_group.add_argument('--validate-only', action='store_true',
                              help='Validate canonical runtime config and inputs without executing providers.')
    

    ## parameter for preprocess
    parser.add_argument('--text_prompt', type=str,
                        help='Text prompt for the scene.')
    parser.add_argument('--img_prompt_folder', type=str,
                        default='modules/preprocess_utils/img_utils',
                        help='Folder of the template text prompt for the scene or the objects.')
    parser.add_argument('--items', type=str, nargs='+',
                        help='List of items to be included in the scene.')
    parser.add_argument('--ref_image_folder', type=str, default='data/ref_img',
                        help='Folder containing reference images for the scene.')
    parser.add_argument('--ref_image_candidate_folder', type=str, default='data/ref_img_candidate',
                        help='Folder containing reference images candidate for the scene.')
    parser.add_argument('--ref_image_num', type=int, default = 1,
                        help='Number of reference images to generate for the scene.')
    parser.add_argument('--vqas_env_path', type=str, default='/home/guyinglin/miniconda3/envs/t2v/bin/python3',
                        help='Path to the vqas environment python executable.')
    parser.add_argument('--vqa_score_folder', type=str, default='data/vqa_score',
                        help='Folder to save the original vqa scores for the images.')
    parser.add_argument('--depth_folder', type=str, default='data/depth',
                        help='Folder to save and load the depth data.')
    parser.add_argument('--seg_folder', type=str, default='data/seg',
                        help='Folder to save and load the segmentation data.')
    parser.add_argument('--items_folder', type=str, default='data/items',
                        help='Folder to save and load the item data in the scene.')
    parser.add_argument('--gpt_prompt_folder', type=str, default='modules/preprocess_utils/gpt_utils',
                        help='Folder containing the GPT prompt files.')
    parser.add_argument('--gpt_apikey_path', type=str, default='modules/preprocess_utils/gpt_utils/apikey.txt',
                        help='Path to the GPT API key file.')
    parser.add_argument('--descrip_folder', type=str, default='data/descrip',
                        help='Folder to save and load the object description data.')
    parser.add_argument('--ref_image_obj_folder', type=str, default='data/ref_img_obj',
                        help='Folder to save and load the reference image for object data.')
    parser.add_argument('--ref_image_obj_candidate_folder', type=str, default='data/ref_img_obj_candidate',
                        help='Folder to save and load the reference image candidate for object data.')
    parser.add_argument('--raw_obj_folder', type=str, default='data/raw_obj',
                        help='Folder to save and load the raw object data.')
    parser.add_argument('--organized_obj_folder', type=str, default='data/organized_obj',
                        help='Folder to save and load the organized object data.')
    parser.add_argument('--layout_front_num', type = int, default = 5, 
                        help = 'When computing the layout, the number of points to compute the front mean position')
    parser.add_argument('--layout_folder', type = str, default = 'data/layout',
                        help='Folder to save and load the layout data.')
    parser.add_argument('--low_poly_folder', type=str, default='data/low_poly',
                        help='Folder to save and load the low poly object data.')
    parser.add_argument('--low_poly_psample', type=int, default = 300000,
                        help='Number of points to sample for reconstructing low poly objects.')
    parser.add_argument('--low_poly_fnum', type=int, default = 2000,
                        help='Number of target faces for the low poly objects.')
    parser.add_argument('--size_folder', type=str, default='data/size',
                        help='Folder to save and load the size data.')
    parser.add_argument('--contain_folder', type=str, default='data/contain',
                        help='Folder to save and load the contain information in the scene.')
    parser.add_argument('--contain_on_folder', type=str, default='data/contain_on',
                        help='Folder to save and load the contain on information in the scene.')
    parser.add_argument('--manifold_code_path', type=str, default='extern/Manifold/build',
                        help='Path to the manifold code folder.')

    ## bbox arranger
    parser.add_argument('--bbox_put_folder', type=str, default='data/bbox_put',
                        help='Folder to save and load the bounding box placement data.')
    parser.add_argument('--max_layer_num', type = int, default = 6,
                        help='The maximum number of layers for the bounding box placement.')
    parser.add_argument('--xyz_order_folder', type=str, default='data/xyz_order',
                        help='Folder to save and load the xyz order data.')
    parser.add_argument('--layer_folder', type=str, default='data/layer',
                        help='Folder to save and load the layer data.')
    parser.add_argument('--forward_sim_layer_layout_folder', type=str, default='data/forward_sim_layer_layout',
                        help='Folder to save and load the layer layout data for forward simulation.')
    parser.add_argument('--high_layer_layout_folder', type=str, default='data/high_layer_layout',
                        help='Folder to save and load the high layer layout data.')
    parser.add_argument('--diff_sim_layer_layout_folder', type=str, default='data/diff_sim_layer_layout',
                        help='Folder to save and load the low layer layout data for differentiable simulation.')
    parser.add_argument('--maintain_xz', action='store_true',
                        help='Solve the xz projection for the bounding box placement.')
    parser.add_argument('--dup', action='store_true',
                        help='Duplicate the items in the scene.')
    parser.add_argument('--duplicate_folder', type=str, default='data/duplicate',
                        help='Folder to save and load the duplicate item information.')
    parser.add_argument('--in_container', action='store_true',
                        help='Use the in_container information for the bounding box placement.')
    parser.add_argument('--ground_y_value_bbox', type = float, default = -0.999,
                        help='The y value of the ground plane when putting the bounding box.')
    parser.add_argument('--gap_y_value', type = float, default = 0.005,
                        help='The gap value for the y axis when putting the bounding box.')
    parser.add_argument('--total_xz_optim_iter', type = int, default = 1000,
                        help='The total number of iterations for the xz optimization.')

                        
    ## physical optimization
    parser.add_argument('--gui_flag', action='store_true',
                        help='Use GUI for visualization during optimization.')
    parser.add_argument('--total_random_attempt', type = int, default = 3,
                        help='Total number of random attempts for physical optimization.')
    parser.add_argument('--total_opt_epoch', type = int, default = 5,
                        help='Total number of epochs for physical optimization.')
    parser.add_argument('--phys_result_folder', type = str, default = '_phys_result',
                        help='Folder to save the results of physical optimization.')
    parser.add_argument('--time_step', type = float, default = 0.03,
                        help='Timestep for the physical simulation.')
    parser.add_argument('--tol_rate', type = float, default = 1e-3,
                        help='Tolerance rate for the linear system.')
    parser.add_argument('--rho', type = float, default = 1000.0)


    parser.add_argument('--ground_y_value', type = float, default = -1.05,
                        help='The y value of the ground plane.')
    parser.add_argument('--end_frame', type = int, default = 2000,
                        help='The end frame for the simulation. Could also be set to be 300.')
    parser.add_argument('--offset_frame', type = int, default = 20,
                        help='The offset frame for the optimization. This is the number of frames to be skipped before the optimization starts.')
    parser.add_argument('--optim_frame_interval', type = int, default = 10,
                        help='The interval frames for the optimization. This is the number of frames to be skipped between each optimization step.')
    parser.add_argument('--weight_y', type = float, default = 20.0,
                        help='The weight for the y axis. This is used to control the weight of difference on the y axis in the optimization.')
    parser.add_argument('--weight_x', type = float, default = 1.0,
                        help='The weight for the x axis. This is used to control the weight of difference on the x axis in the optimization.')
    parser.add_argument('--weight_z', type = float, default = 1.0,
                        help='The weight for the z axis. This is used to control the weight of difference on the z axis in the optimization.')
    parser.add_argument('--phys_lr', type = float, default = 0.1,
                        help='Learning rate for the physical optimization.')
                    
                    

    
    ## visualization 
    parser.add_argument('--end_frame_upper_bound', type = int, default = 300,
                        help='The upper bound for the end frame. This is used to control the maximum number of frames for the simulation.')
    parser.add_argument('--low_poly', action='store_true',
                        help='Use low poly version of the geometry.')
    parser.add_argument('--semantic_optim_output_folder', type=str, default='vis/scene_high_poly',
                        help='Folder to save the semantic optimization output.')
    parser.add_argument('--simulation_output_folder', type=str, default='vis/simulated_scene',
                        help='Folder to save the forward simulation output.')
    parser.add_argument('--optim_attempt_num', type = int, default = 0,
                        help='The attempt number of the optimization, 0 means the initial one without optimization.')
    parser.add_argument('--optim_initial_scene_folder', type=str, default='visualize/optim_s0',
                        help='Folder to save the optimized initial scene.')
    
    # Architecture for network
    net_group = parser.add_argument_group('net')

    # Arguments for dataset
    data_group = parser.add_argument_group('dataset')
    data_group.add_argument('--simu_save_dir', type = str, default = 'output',
                            help = 'The directory for saving the simulation results.')
    data_group.add_argument('--max_intersection_iter', type = int, default = 30,
                            help = 'The max interation number for solving the intersections.')
    data_group.add_argument('--object_sample_num', type = int, default = 10000,
                            help = 'The sampling number for each object in order to get the sdf query function.')
    data_group.add_argument('--protect_strip_value', type = float, default = 0.008,
                            help = 'The protection strip for more efficient simulation.')
    data_group.add_argument('--global_min_x', type = float, default = -3,
                            help = 'The global min x value for the bounding box.')
    data_group.add_argument('--global_min_y', type = float, default = -1,
                            help = 'The global min y value for the bounding box.')  
    data_group.add_argument('--global_min_z', type = float, default = -3,
                            help = 'The global min z value for the bounding box.')
    data_group.add_argument('--global_max_x', type = float, default = 3,
                            help = 'The global max x value for the bounding box.')
    data_group.add_argument('--global_max_y', type = float, default = 4,
                            help = 'The global max y value for the bounding box.')
    data_group.add_argument('--global_max_z', type = float, default = 3,
                            help = 'The global max z value for the bounding box.')



    # Arguments for optimizer
    optim_group = parser.add_argument_group('optimizer')
    optim_group.add_argument('--lr', type = float, default = 0.01, 
                             help = 'Learning rate.')
    optim_group.add_argument('--lr_step_size', type = int, default = 1, 
                             help = 'Step size for learning rate decay.')
    optim_group.add_argument('--lr_step_gamma', type = float, default = 0.999, 
                             help = 'Gamma for learning rate decay.')
        

 
    # Parse and run
    if return_parser:
        return parser
    else:
        return argparse_to_str(parser, argv=argv)


def argparse_to_str(parser, argv=None):

    args = parser.parse_args(argv)

    args_dict = {}
    for group in parser._action_groups:
        group_dict = {a.dest:getattr(args, a.dest, None) for a in group._group_actions}
        args_dict[group.title] = vars(argparse.Namespace(**group_dict))

    pp = pprint.PrettyPrinter(indent=2)
    args_str = pp.pformat(args_dict)
    args_str = f'```{args_str}```'

    return args, args_str
