import os
import json
from pat3d.preprocessing.gpt import query_gpt4o


def get_size(self, args):
    ## mkdir the size folder if not exist
    if not os.path.exists(args.size_folder):
        os.makedirs(args.size_folder)

    ## load the items in the scene 
    item_json_path = f'{args.items_folder}/{self.scene_name}.json'
    with open(item_json_path, "r") as file:
        item_json_result = json.load(file)
    item_list = list(item_json_result.keys())
    
    ## get the path of the ref scene image 
    ref_img_path = f'{args.ref_image_folder}/{self.scene_name}.png'

    ## load the text prompt for size query 
    text_prompt_path = os.path.join(args.gpt_prompt_folder, 'get_size.txt')
    with open(text_prompt_path, "r") as file:
        text_prompt = file.read().strip()

    ## add the items in the scene to the text prompt
    additional_text = f'This scene contains '
    for item_name in item_list:
        additional_text += f'{item_name}, '
    additional_text = additional_text[:-2] + '. '
    final_text_prompt = additional_text + text_prompt

    ## query gpt for the size information
    size_info = query_gpt4o(ref_img_path, args.gpt_apikey_path, query_prompt = final_text_prompt)
    print("Size information: ", size_info)
    save_size_path = f'{args.size_folder}/{self.scene_name}.json'
    with open(save_size_path, "w") as file:
        json.dump(size_info, file)
