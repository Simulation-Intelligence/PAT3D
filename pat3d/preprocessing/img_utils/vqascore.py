import sys 
import os
import argparse
import json
import numpy as np


current_path = (os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
sys.path.append(current_path)

import extern.t2v_metrics.t2v_metrics as t2v_metrics 


def parse_args():
    parser = argparse.ArgumentParser(description='Compute VQA score')
    parser.add_argument('--text_prompt', type = str, nargs = '+', default = [], help = 'List of the text prompts')
    parser.add_argument('--image_paths', type = str, nargs = '+', default = [], help = 'List of image paths')
    parser.add_argument('--img_prompt_folder', type = str, default = '', help = 'Folder path for the criteria text prompts')
    parser.add_argument('--vqa_score_path', type = str, default = '', help = 'Path to save the score matrix')
    parser.add_argument('--img_final_folder', type = str, default = '', help = 'Path to save the final image')
    parser.add_argument('--img_name', type = str, default = '', help = 'Scene or object name')
    parser.add_argument('--category', type = str, default = 'scene', help = 'Whether the input is for a scene or object, default is scene')

    args = parser.parse_args()
    return args


def get_score_matrix(text_prompt_list, image_paths_list):

    #clip_flant5_score = t2v_metrics.VQAScore(model='clip-flant5-xxl') # our recommended scoring model
    clip_flant5_score = t2v_metrics.VQAScore(model='clip-flant5-xl') 
    scores = clip_flant5_score(images = image_paths_list, texts = text_prompt_list) # scores[i][j] is the score between image i and text j
    
    return scores.cpu().numpy()

def save_scores_to_file(scores, image_paths_list, save_score_path):
    ## write the scores for each image as the list in the json dict 
    result_dict = {}
    for i, score in enumerate(scores):
        image_path = image_paths_list[i]
        result_dict[image_path] = list(float(score_piece) for score_piece in score)

    if not os.path.exists(os.path.dirname(save_score_path)):
        os.makedirs(os.path.dirname(save_score_path))

    with open(save_score_path, 'w') as f:
        json.dump(result_dict, f, indent=4)


def load_additional_prompt(prompt_path, return_list = False):

    with open(prompt_path, 'r') as file:
        lines = file.readlines()
        if return_list:
            additional_prompt = [line.strip() for line in lines]
        else:
            additional_prompt = " ".join(line.strip() for line in lines)

    return additional_prompt

if __name__ == "__main__":

    args = parse_args()

    original_text_prompt = args.text_prompt

    if args.category == 'object':
        criteria_path = f'{args.img_prompt_folder}/obj_criteria.txt'
    else:
        criteria_path = f'{args.img_prompt_folder}/scene_criteria.txt'

    additional_criteria_text_list = load_additional_prompt(criteria_path, return_list = True)
    criteria_text_list = original_text_prompt + additional_criteria_text_list

    scores = get_score_matrix(criteria_text_list, args.image_paths)

    ## save the original scores
    save_scores_to_file(scores, args.image_paths, args.vqa_score_path)
    
    ## compute the weighted sum of the scores
    if args.category == 'object':
        weights = np.array([2, 1, 1], dtype=np.float32)
    else:
        weights = np.array([2, 1, 1, 1], dtype=np.float32)
    weighted_scores = np.sum(scores * weights, axis=1)

    ## pick the best image based on the weighted scores
    best_image_index = np.argmax(weighted_scores)
    best_image_path = args.image_paths[best_image_index]
    best_image_save_path = f'{args.img_final_folder}/{args.img_name}.png'
    print(f'Best image path: {best_image_path}')
    
    ## copy the best image to the final folder
    if not os.path.exists(args.img_final_folder):
        os.makedirs(args.img_final_folder)
    os.system(f'cp {best_image_path} {best_image_save_path}')



    
