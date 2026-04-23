import os
import sys
import glob
import multiprocessing
import tqdm
import random
import threading
import torch


blender_path = '/usr/local/blender/blender'

def render_multiview(source_folder, target_folder, gpu_id=0):
    # Get all the glb files in the source folder
    obj_filenames = glob.glob(os.path.join(source_folder, '*.glb')) + glob.glob(os.path.join(source_folder, '*.obj'))

    for obj_filename in tqdm.tqdm(obj_filenames):
        output_filename = os.path.join(target_folder, os.path.basename(obj_filename).split('.')[0])
        # Process each object
        torch.cuda.empty_cache()
        # first render a front view
        cmd = f'/usr/local/blender/blender --background --python ./blender_script.py -- --object_path {obj_filename}  --output_dir {output_filename}'
        cmd = f"export DISPLAY=:0.{gpu_id} && {cmd}"
        os.system(cmd)


def render_multiview_multiobjs(source_folder, target_folder, gpu_id=0):
    # Get all the glb files in the source folder
    obj_foldernames = glob.glob(os.path.join(source_folder, '*'))
    
    #for obj_foldername in tqdm.tqdm(obj_foldernames):
    
    obj_folder_name = source_folder
    output_filename = os.path.join(target_folder, os.path.basename(source_folder).split('.')[0])
    # Process each object
    torch.cuda.empty_cache()
    # first render a front view
    # cmd = f'/data1/users/yuanhao/blender-4.0.0-linux-x64/blender --background --python ./blender_script.py -- --object_path {obj_filename}  --output_dir {output_filename}'
    cmd = f'{blender_path} --background --python ./blender_script_multi_objs.py -- --object_path {source_folder}  --output_dir {output_filename}'
    cmd = f"export DISPLAY=:0.{gpu_id} && {cmd}"
    os.system(cmd)


if __name__ == "__main__":

    gpu_id = 2

    # source_parent_folder = '/data1/users/yuanhao/guying_proj/eval/baselines'
    # target_parent_folder = '/data1/users/yuanhao/guying_proj/eval/baselines_renders_2k'
    # # target_folder = '/data1/datasets/garment-data/image-data/test'


    source_parent_folder = '/media/guyinglin/yihao/guying_sim_gen/mask_loss/vis/more_examples'
    target_parent_folder = '/media/guyinglin/yihao/guying_sim_gen/mask_loss/vis/figures/more_examples'
    baseline_names = ['stackedeggs_optim_0']
    
    for name in baseline_names:
        source_folder = os.path.join(source_parent_folder, name)
        target_folder = os.path.join(target_parent_folder, name)
        os.makedirs(target_folder, exist_ok=True)

        render_multiview_multiobjs(source_folder, target_folder, gpu_id=gpu_id)
        #render_multiview(source_folder, target_folder, gpu_id=gpu_id)


    
    