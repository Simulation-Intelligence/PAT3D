import os
import json
import glob
from pat3d.preprocessing.gpt import query_gpt4o

def get_contain_info(args, scene_name):

    ## load the description json file
    descrip_json_path = f'{args.descrip_folder}/{scene_name}.json'
    with open(descrip_json_path, "r") as file:
        descrip_json_result = json.load(file)
    item_list = list(descrip_json_result.keys())

    ## build gpt prompt 
    text_prompt_path = os.path.join(args.gpt_prompt_folder, 'get_contain.txt')
    with open(text_prompt_path, "r") as file:
        text_prompt = file.read().strip()

    ## add the items in the scene to the text prompt
    additional_text = f'This scene contains objects named '
    for item_name in item_list:
        additional_text += f'{item_name}, '
    additional_text = additional_text[:-2] + '. '
    final_text_prompt = additional_text + text_prompt

    #print(f"Final text prompt: {final_text_prompt}")
    #exit(0)

    ## get the path of the ref scene image 
    ref_img_path = None
    for img_file in os.listdir(args.ref_image_folder):
        if scene_name in img_file:
            ref_img_path = f'{args.ref_image_folder}/{img_file}'
            break
    if ref_img_path is None:
        print("Image not found")
        exit(0)

    ## get the object description
    contain_info = query_gpt4o(ref_img_path, args.gpt_apikey_path, query_prompt = final_text_prompt)
    
    ## create the object description folder if not exist
    if not os.path.exists(args.contain_folder):
        os.makedirs(args.contain_folder)

    ## save the object description json file
    save_descrip_path = f'{args.contain_folder}/{scene_name}.json'
    with open(save_descrip_path, "w") as file:
        json.dump(contain_info, file)


def get_contain_on_info(args, scene_name):

    ## load the description json file
    descrip_json_path = f'{args.descrip_folder}/{scene_name}.json'
    with open(descrip_json_path, "r") as file:
        descrip_json_result = json.load(file)
    item_list = list(descrip_json_result.keys())

    ## build gpt prompt 
    text_prompt_path = os.path.join(args.gpt_prompt_folder, 'get_contain_on.txt')
    with open(text_prompt_path, "r") as file:
        text_prompt = file.read().strip()

  ## add the items in the scene to the text prompt
    additional_text = f'This scene contains objects named '
    for item_name in item_list:
        additional_text += f'{item_name}, '
    additional_text = additional_text[:-2] + '. '
    final_text_prompt = additional_text + text_prompt

    #print(f"Final text prompt: {final_text_prompt}")
    #exit(0)

    ## get the path of the ref scene image 
    ref_img_path = None
    for img_file in os.listdir(args.ref_image_folder):
        if scene_name in img_file:
            ref_img_path = f'{args.ref_image_folder}/{img_file}'
            break
    if ref_img_path is None:
        print("Image not found")
        exit(0)

    ## get the object description
    contain_info = query_gpt4o(ref_img_path, args.gpt_apikey_path, query_prompt = final_text_prompt)
    
    ## create the object description folder if not exist
    if not os.path.exists(args.contain_on_folder):
        os.makedirs(args.contain_on_folder)

    ## save the object description json file
    save_descrip_path = f'{args.contain_on_folder}/{scene_name}.json'
    with open(save_descrip_path, "w") as file:
        json.dump(contain_info, file)
