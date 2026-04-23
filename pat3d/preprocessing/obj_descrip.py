import os
import json
import glob
from time import sleep
from pat3d.preprocessing.gpt import query_gpt4o

def get_obj_descrip(args, scene_name):

    ## load segmented images 
    seg_img_folder = f'{args.seg_folder}/{scene_name}'
    seg_img_path_list = glob.glob(f'{seg_img_folder}/*')

    ## filter the scene image and image with ann.png as the suffix
    seg_img_path_list = [img_path for img_path in seg_img_path_list if not(f'{scene_name}_segmentation.png' in img_path) and not img_path.endswith('ann.png')]

    ## build gpt prompt 
    text_prompt_path = os.path.join(args.gpt_prompt_folder, 'get_obj_descrip.txt')

    object_descrip_dict = {}
    ## process each image 
    query_count = 0
    for seg_img_path in seg_img_path_list:

        ## get the image name and the object name
        full_object_name = seg_img_path.split('/')[-1].split('_')[-2]
        object_name = ''.join([i for i in full_object_name if not i.isdigit()])
        
        ## get the text prompt
        with open(text_prompt_path, "r") as file:
            text_prompt = file.read().strip()
        add_object_identify = f'The object is {object_name}. '
        text_prompt = add_object_identify + text_prompt

        ## get the object description
        object_descrip_piece = query_gpt4o(seg_img_path, args.gpt_apikey_path, query_prompt = text_prompt)

        ## sleep for a while to avoid gpt api limit
        if query_count%5 == 0 and query_count != 0:
            print('Sleeping for 60 seconds to avoid GPT rate limit per minute.')
            sleep(61)

        for key in object_descrip_piece.keys():
            object_descrip_piece = object_descrip_piece[key]
        
        ## get the 
        object_descrip_dict[full_object_name] = add_object_identify + object_descrip_piece
        
        query_count += 1

    ## create the object description folder if not exist
    if not os.path.exists(args.descrip_folder):
        os.makedirs(args.descrip_folder)

    ## save the object description json file
    save_descrip_path = f'{args.descrip_folder}/{scene_name}.json'
    with open(save_descrip_path, "w") as file:
        json.dump(object_descrip_dict, file)
