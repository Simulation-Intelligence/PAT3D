import os
import subprocess
import sys
from pat3d.preprocessing.gpt import query_gpt4o_item


def get_items(scene_name, image_folder, gpt_prompt_folder, gpt_apikey_path):

    ## load the gpt text prompt file 
    text_prompt_path = os.path.join(gpt_prompt_folder, 'get_items.txt')
    
    ## load the image with scene_name as the file name but any suffix is fine
    img_path = None
    for img_file in os.listdir(image_folder):
        if scene_name in img_file:
            img_path = f'{image_folder}/{img_file}'
            break
    if img_path is None:
        print("Image not found")
        return

    result = query_gpt4o_item(img_path, gpt_apikey_path, text_prompt_path)

    item_list = []
    for object_name in result.keys():
        object_num = result[object_name]
        for i in range(object_num):
            item_list.append(object_name)

    return item_list, result





    

    ## return a list that list all the items in the scene
    ## for those duplicated items, we list all of them 

    



def organize_items(seg_folder, scene_name):
    ## check the input folder to find the image with the name image_name
    seg_folder_path = os.path.join(seg_folder, scene_name)
    seg_files = os.listdir(seg_folder_path)
    object_num_dict = {}
    for seg_file in seg_files:
        if (seg_file.endswith('ann.png')):
            ann_file_path = seg_file
            img_file_path = seg_file.replace('_ann.png', '.png')
            object_name = seg_file.split('_')[-3]

            ## adjust the number of objects
            if object_name not in object_num_dict:
                object_num_dict[object_name] = 1
            else:
                object_num_dict[object_name] += 1
            
            ## rename the image file 
            object_rank = object_num_dict[object_name]
   
            new_ann_file_path = f'{scene_name}_{object_name}{object_rank}_{object_name}{object_rank}_ann.png'
            new_img_file_path = f'{scene_name}_{object_name}{object_rank}_{object_name}{object_rank}.png'
            os.rename(os.path.join(seg_folder_path, ann_file_path), os.path.join(seg_folder_path, new_ann_file_path))
            os.rename(os.path.join(seg_folder_path, img_file_path), os.path.join(seg_folder_path, new_img_file_path))


def check_seg(seg_folder, image_name, object_name_list):
    ## load all the segmented files in the folder 
    seg_folder_path = os.path.join(seg_folder, image_name)
    seg_files = os.listdir(seg_folder_path)
    seg_object_name_list = []
    seg_flag = True
    for seg_file in seg_files:
        if (seg_file.endswith('ann.png')):
            ## get the two object names
            #fruitbasket_1_banana_banana.png
            object_name_1 = seg_file.split('_')[-3]
            object_name_2 = seg_file.split('_')[-2]
            if not(object_name_1 == object_name_2):
                print('Warning: wrong segmentation for', seg_file)
                seg_flag = False
            else:
                seg_object_name_list.append(object_name_1)
    
    ## compare two lists
    if set(seg_object_name_list) != set(object_name_list):
        print('Warning: segmentation object list is not equal to the input object list')
        seg_flag = False
   
    return seg_flag

    

def get_seg(image_name, image_folder, object_name_list, output_dir):

    ## check the input folder to find the image with the name image_name
    #input_img_path = None
    #for img_path in os.listdir(image_folder):
    #    if image_name in img_path:
    #        input_img_path = f'{image_folder}/{img_path}'
    input_img_path = os.path.join(image_folder, f'{image_name}.png')

    if input_img_path is None:
        print("Image not found")
        return

    ## check if the output directory exists, if not, create it
    seg_output_dir_path = os.path.join(output_dir, image_name)
    if os.path.exists(seg_output_dir_path):
        rm_command = 'rm -r ' + seg_output_dir_path
        os.system(rm_command)
    os.makedirs(seg_output_dir_path)

    # Run the python command
    subprocess.run([sys.executable, 'extern/sfg2/seg.py', '--input', input_img_path, '--seg_output', seg_output_dir_path, '--object_name_list', *object_name_list], check=True)
