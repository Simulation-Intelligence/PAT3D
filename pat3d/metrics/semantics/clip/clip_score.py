import torch
from torchmetrics.functional.multimodal import clip_score
from torchvision import transforms
from PIL import Image
import os
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import json


def get_prompts():
    prompts = [
        'a cozy cartoon setup with a bed, beside table, lamp, bookshelf, and a small dog on the bed.',
        'a luxurious CG scene with a couch, coffee table, small art piece, a cozy rug.',
        'a warm setup with a bed, bedside tables, plant, a few books, and a small decorative lamp.',
        'a baby bunny sitting on top of a stack of pancakes',
        'a Wizard standing in front of a Wooden Desk, gazing into a Crystal Ball perched atop the Wooden Desk, with a Stack of Ancient Spell Books perched atop the Wooden Desk',
        'a blue jay standing on a large basket of rainbow macarons',
        'On a table, there is a vase with a bouquet of flowers. Beside it, there is a plate of cake',
        'Horizontal perspective, showing the complete objects. In a basket of fruit, there are 5 fruits.',
        'Horizontal perspective, showing the complete objects. Four stacked cups and four stacked plates.',
        'A white ceramic cup holds two blue and green toothbrushes, a blue and white toothpaste tube, and a blue-handled razor, while a white soap dish beside it contains two bars of soap—one white and one beige.',
        'A vintage wooden radio with a small cow figurine on top sits on a stack of three hardcover books, next to a wooden cup holding colorful pencils.',
        'A brown leather sofa decorated with plush toys, including a large teddy bear, a gray elephant, a white rabbit, a yellow giraffe, and two throw pillows, sits in a cozy room with two round burgundy floor cushions in front.',
        'A round blue inflatable pool filled with small blue, white, and turquoise plastic balls features a toy sailboat, two yellow rubber ducks, a green turtle, and an orange starfish.',
        'A wooden table with a red fire extinguisher sits in front of a metal shelving unit with a gray cap on one of the shelves, flanked by large cardboard sheets on one side and a wheeled cart holding potted green plants on the other.',
        'A stack of colorful wooden blocks arranged vertically, featuring red, blue, yellow, green, orange, and purple pieces, balanced on a flat surface.',
        'a cute plush toy dog with five different colored hats stacked on its head'
    ]
    return prompts

@torch.no_grad()
def calculate_per_image_score(img_path, prompt):
    transform = transforms.Compose([
        transforms.ToTensor(),           # float32 [0,1]
    ])
    pil = Image.open(img_path).convert("RGB")
    img = transform(pil)                   # (3,224,224), float32
    img_uint8 = (img * 255).to(torch.uint8).unsqueeze(0).to(device)  # (1,3,224,224)
    score = clip_score(img_uint8, [prompt], model_name_or_path="openai/clip-vit-base-patch16").item()
    return score

if __name__ == "__main__":

    image_dir = "/media/guyinglin/yihao/guying_sim_gen/mask_loss/rebuttal/all_renderings/comparison_render"
    prompts = get_prompts()
    #baseline_methods = ['blenderMCP', 'graphdreamer', 'pat3d', 'midi']
    #baseline_methods = ['blenderMCP']
    #baseline_methods = ['graphdreamer']
    #baseline_methods = ['pat3d']
    baseline_methods = ['midi']
    results = {}
    for baseline_method in baseline_methods:

        results[baseline_method] = {}   
        ## list all the scenes in the image_dir
        scene_cases = [f for f in os.listdir(os.path.join(image_dir, baseline_method))]
        scene_cases.sort()
        ## remove the '.DS_Store' and 'score.json
        scene_cases = [case for case in scene_cases if case != '.DS_Store' and case != 'scores.json']
        
        for scene_case in scene_cases:
            results[baseline_method][scene_case] = {}
            scene_id = int(scene_case.split('_')[0]) - 1
            prompt = prompts[scene_id]
            #print(scene_case, '    ', prompt)
            ## load all the images under the scene_case folder 
            scene_folder = os.path.join(image_dir, baseline_method, scene_case)
            if baseline_method == 'graphdreamer':
                scene_folder = os.path.join(scene_folder, 'it10000-test-G')
            
            ## load all the images under the scene_folder
            image_files = [f for f in os.listdir(scene_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            image_files.sort()
            scene_score_list = []
            for img_file in image_files:
                score = calculate_per_image_score(os.path.join(scene_folder, img_file), prompt)
                #print(f"{baseline_method} {scene_case} {img_file} ⟶ \"{prompt}\": {score:.4f}")
                scene_score_list.append(score)

            scene_score_mean = np.mean(scene_score_list)
            results[baseline_method][scene_case]['list'] = scene_score_list
            results[baseline_method][scene_case]['mean']= scene_score_mean
            print(f"{baseline_method} {scene_case} mean score: {scene_score_mean:.4f}")        
        print(results)
        json.dump(results, open(f'clip_score_{baseline_method}.json', 'w'))

            
            

            
