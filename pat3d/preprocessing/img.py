import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import sys 
import json
import time 
import shutil
from PIL import Image
import subprocess


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from pat3d.preprocessing.img_utils.query_reve import query_status, fetch_img_urls, download_image, query_reve_img, load_apikey






def resize_image(save_img_path):
    """
    Resize an image to a specified size and save it to a new path.
    
    Args:
        save_img_path (str): The path to the image file.
        
    Returns:
        None: The resized image is saved to the specified path.
    """
    '''
    # Use glob to find files matching the pattern
    matching_files = glob.glob(save_img_path)
    if matching_files:
        # Take the first matching file
        image_path = matching_files[0]
        scene_image = Image.open(image_path)  # Use PIL to load the image
    else:
        print(f"No image found for pattern: {save_img_path}")
        return  # Exit the function if no image is found
    '''

    scene_image = Image.open(save_img_path)

    # Get the original width and height
    width, height = scene_image.size

    # Resize the image so that the dimensions can be divided by 8
    new_width = (width // 8) * 8
    new_height = (height // 8) * 8

    # Ensure the dimensions are shorter than 2048
    max_size = 2048
    if new_width > max_size or new_height > max_size:
        ratio = min(max_size / new_width, max_size / new_height)
        new_width = int(new_width * ratio)
        new_height = int(new_height * ratio)

    # Resize the image while keeping the aspect ratio
    resized_image = scene_image.resize((new_width, new_height), Image.Resampling.LANCZOS)  # Use Image.LANCZOS if Pillow < 10.0.0
    
    # Save the resized image
    #output_path = "resized_image.png"  # You can specify a different output path
    resized_image.save(save_img_path)

    #print(f"Image resized and saved to: {output_path}")

def generate_scene_candidates(additional_text_prompt_folder, text_prompt, ref_image_candidate_folder, scene_name, ref_image_num):
 
    additional_scene_prompt = load_additional_prompt(f'{additional_text_prompt_folder}/additional_scene_prompt.txt')

    full_text_prompt = f"{additional_scene_prompt} {text_prompt}"

    ## create the image folder for storing all the candidate scene images
    img_save_folder = f'{ref_image_candidate_folder}/{scene_name}'
    if os.path.exists(img_save_folder):
        shutil.rmtree(img_save_folder)
    os.makedirs(img_save_folder)

    ## generate the candidate scene image 
    img_save_path_prefix = f'{ref_image_candidate_folder}/{scene_name}'
    api_key_path = f'{additional_text_prompt_folder}/reve_apikey.txt'
    generate_scene_image(full_text_prompt, img_save_path_prefix, api_key_path, ref_image_num, scene_name)




def generate_scene_image(text_prompt, img_save_path_prefix, api_key_path, img_num, scene_name):
    
    for img_count in range(img_num):
        img_save_path = f'{img_save_path_prefix}/{scene_name}_{img_count}.png' 
        api_key = load_apikey(api_key_path)

        while True:
            try:
                out_json = query_reve_img(text_prompt, api_key, img_num)
                #print('out_json:', out_json)
                result_url = query_status(out_json, api_key)
                #print('result_url:', result_url)
                img_urls = fetch_img_urls(result_url, api_key)['output']
                download_image(img_urls, img_save_path)
                break
            except:
                print('out_json:', out_json)
                print('sleep for 42 seconds')
                time.sleep(42)    



def select_best_object_image(obj_description_dict, img_candidate_folder, img_folder, scene_name, img_prompt_folder, vqas_env_path, vqa_score_folder):
    for shape_name in obj_description_dict.keys():

        ## load all the images path list in the img_candidate_folder
        img_candidate_case_folder = f'{img_candidate_folder}/{scene_name}/{shape_name}'
        img_candidate_path_list = glob.glob(f'{img_candidate_case_folder}/{shape_name}_*.png')

        ## image target folder 
        img_target_folder = f'{img_folder}/{scene_name}'

        ## get additional prompt for the object
        description_prompt = obj_description_dict[shape_name]

        ## get the weighted scores for each image and save the original scores  
        save_obj_score_path = f'{vqa_score_folder}/{scene_name}_object'
        if not os.path.exists(save_obj_score_path):
            os.makedirs(save_obj_score_path)
        save_score_path = f'{save_obj_score_path}/{scene_name}_{shape_name}.json'
        subprocess.run([vqas_env_path, "pat3d/preprocessing/img_utils/vqascore.py", "--image_paths", *img_candidate_path_list, \
                        "--text_prompt", description_prompt, "--img_prompt_folder", img_prompt_folder, \
                        "--vqa_score_path", save_score_path, '--img_final_folder', img_target_folder, \
                        "--img_name", shape_name, "--category", 'object'], check=True)
    

    return 

def generate_object_image(text_prompt, img_save_path_prefix, api_key_path, img_num, object_name):
    
    for img_count in range(img_num):
        img_save_path = f'{img_save_path_prefix}/{object_name}_{img_count}.png' 
        api_key = load_apikey(api_key_path)

        while True:
            try:
                out_json = query_reve_img(text_prompt, api_key, img_num)
                #print('out_json:', out_json)
                result_url = query_status(out_json, api_key)
                #print('result_url:', result_url)
                img_urls = fetch_img_urls(result_url, api_key)['output']
                download_image(img_urls, img_save_path)
                break
            except:
                print('out_json:', out_json)
                print('sleep for 43 seconds')
                time.sleep(43)    

def load_additional_prompt(prompt_path, return_list = False):

    with open(prompt_path, 'r') as file:
        lines = file.readlines()
        if return_list:
            additional_prompt = [line.strip() for line in lines]
        else:
            additional_prompt = " ".join(line.strip() for line in lines)

    return additional_prompt


def select_best_scene_image(img_candidate_folder, img_folder, scene_name, img_prompt_folder, text_prompt, vqas_env_path, vqa_score_folder):
    
    ## load all the images path list in the img_candidate_folder
    img_candidate_case_folder = f'{img_candidate_folder}/{scene_name}'
    img_candidate_path_list = glob.glob(f'{img_candidate_case_folder}/*.png')

    ## get the weighted scores for each image and save the original scores  
    save_score_path = f'{vqa_score_folder}/{scene_name}.json'
    subprocess.run([vqas_env_path, "pat3d/preprocessing/img_utils/vqascore.py", "--image_paths", *img_candidate_path_list, \
                    "--text_prompt", text_prompt, "--img_prompt_folder", img_prompt_folder, "--vqa_score_path", save_score_path, \
                    '--img_final_folder', img_folder, "--img_name", scene_name], check=True)

        
    

    ## combine the image_candidate_path_list to be a 

    

    
    
    # Implement your selection logic here
    pass

if __name__ == "__main__":

    scene_name = "homebookshelf"
    example_text_prompt = "Horizontal perspective, showing the complete objects. A two-tier bookshelf, with 3 decorations on the top shelf and 3 books on the bottom shelf."
    img_num = 3

    ## create the image folder for storing all the candidate images
    img_save_folder = f'data/ref_img_candidate/{scene_name}'
    if os.path.exists(img_save_folder):
        shutil.rmtree(img_save_folder)
    os.makedirs(img_save_folder)
    img_save_path_prefix = f'data/ref_img_candidate/{scene_name}'

    ## load the api key
    api_key_path = 'pat3d/preprocessing/img_utils/reve_apikey.txt'

    ## generate the scene image 
    generate_scene_image(example_text_prompt, img_save_path_prefix, api_key_path, img_num, scene_name)
