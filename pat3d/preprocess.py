from contextlib import redirect_stdout
import json 
import os
import glob
import sys
from time import sleep
import time
import shutil
from pat3d.preprocessing.utils import load_image
from pat3d.preprocessing.depth import get_depth
from pat3d.preprocessing.seg import get_seg, check_seg, organize_items, get_items
from pat3d.preprocessing.gpt import query_gpt4o
from pat3d.preprocessing.obj_gen import generate_obj_items, organize_obj_items, generate_obj_ref_images
from pat3d.preprocessing.layout import get_initial_layout, get_initial_layout_duplicate
from pat3d.preprocessing.reset_y import reset_ground
from pat3d.preprocessing.low_poly import get_low_poly_new
from pat3d.preprocessing.size import get_size
from pat3d.preprocessing.contain import get_contain_on_info
from pat3d.preprocessing.obj_descrip import get_obj_descrip
from pat3d.preprocessing.img import resize_image, generate_scene_candidates, select_best_scene_image, select_best_object_image


class Preprocessor(object):

    def __init__(self, args):
        self.args = args
        self.scene_name = args.scene_name
        self.text_prompt = args.text_prompt
        self.items = args.items
    
    def preprocess(self):
        time_dir = os.path.join("results", "preprocess_timing")
        os.makedirs(time_dir, exist_ok=True)
        time_scene_name = self.scene_name if self.scene_name else "default"
        time_write_file_name = os.path.join(time_dir, f"{time_scene_name}.txt")

        ## first step 
        '''
        self.get_depth()
        self.get_items()
        self.get_segmentation()
        '''

        ## second step 
        '''
        time_start = time.time()
        organize_items(self.args.seg_folder, self.scene_name)
        self.get_object_description()    
        self.get_size_info()
        self.get_contain_on_info()   
        time_end = time.time()
        print(f"Second step: {time_end - time_start} seconds")
        with open(time_write_file_name, 'a') as f:
            f.write(f"Preprocess second step: {time_end - time_start} seconds\n")
        '''
        
        '''
        time_start = time.time()        ## third step 
        self.generate_objects()
        time_end = time.time()
        print(f"Preprocess generation step: {time_end - time_start} seconds")
        with open(time_write_file_name, 'a') as f:
            f.write(f"Preprocess generation step: {time_end - time_start} seconds\n")
        '''
        
        time_start = time.time()
        self.organize_objects()
        self.get_scene_layout()
        self.get_low_poly()
        time_end = time.time()
        print(f"Third step: {time_end - time_start} seconds")
        with open(time_write_file_name, 'a') as f:
            f.write(f"Preprocess third step: {time_end - time_start} seconds\n")
        #'''
             

    def add_duplicate_items(self):

        ## load the duplicate item json file 
        duplicate_info_path = f'{self.args.duplicate_folder}/{self.scene_name}.json'

        ## the duplicate information 
        with open(duplicate_info_path, "r") as file:
            duplicate_info = json.load(file)

        ## load the original item json file
        item_json_path = f'{self.args.items_folder}/{self.scene_name}.json'
        with open(item_json_path, "r") as file:
            item_json_result = json.load(file)
        
        ## load the original obj description json file
        descrip_json_path = f'{self.args.descrip_folder}/{self.scene_name}.json'
        with open(descrip_json_path, "r") as file:
            object_descrip_dict = json.load(file)

        ## organize the item information to get seperate text prompts 
        item_text_prompt_dict = {}
        item_num_dict = {}
        for item_name in duplicate_info.keys():
            color_list = duplicate_info[item_name]["color"]
            shape_prompt = duplicate_info[item_name]["shape"]
            material_prompt = duplicate_info[item_name]["material"]
            orientation_prompt = duplicate_info[item_name]["orientation"]
            item_total_num = item_json_result[item_name]
            for color_id in range(len(color_list)):
                color = color_list[color_id]
                if color_id == (len(color_list) - 1):
                    color_piece_num = item_total_num - (item_total_num // len(color_list)) * (len(color_list) - 1)
                else:
                    color_piece_num = item_total_num // len(color_list)
                color_prompt = f"The color of the {item_name} is {color}."
                total_prompt = f"This is a {item_name}. {color_prompt} {shape_prompt} {material_prompt} {orientation_prompt}"
                item_text_prompt_dict[f'{item_name}{color_id}'] = total_prompt
                item_num_dict[f'{item_name}{color_id}'] = color_piece_num

                object_descrip_dict[f'{item_name}{color_id}'] = total_prompt

        ## save back the item json file
        with open(descrip_json_path, "w") as file:
            json.dump(object_descrip_dict, file)
        
        ## save the item num dict to duplicate folder 
        item_num_path = f'{self.args.duplicate_folder}/{self.scene_name}_item_num.json'
        with open(item_num_path, "w") as file:
            json.dump(item_num_dict, file)
        


    def generate_image(self):
        
        #generate_scene_candidates(self.args.img_prompt_folder, self.args.text_prompt, self.args.ref_image_candidate_folder, self.args.scene_name, self.args.ref_image_num)

        ## select the best image from the candidate images using vqascore 
        #select_best_scene_image(self.args.ref_image_candidate_folder, self.args.ref_image_folder, self.scene_name, self.args.img_prompt_folder, self.args.text_prompt, self.args.vqas_env_path, self.args.vqa_score_folder)
        
        # This function should generate an image based on the text prompt and items
        save_img_path = f'{self.args.ref_image_folder}/{self.scene_name}.png'

        ## resize the image so that the width and height can be divided by 8 and are shorter than 2048
        resize_image(save_img_path)

        ## load the image with any suffix
        self.scene_image = load_image(save_img_path)

      
    def get_depth(self):

        get_depth(self.scene_name, self.args.ref_image_folder, self.args.depth_folder)
        print(f"Depth data saved in {self.args.depth_folder}/{self.scene_name}")


    def get_items(self):

        if self.args.items is None:
            print("Items not provided. Generating items using GPT-4o.")
            self.items, item_json_result = get_items(self.scene_name, self.args.ref_image_folder, self.args.gpt_prompt_folder, self.args.gpt_apikey_path)
        else:
            self.items = self.args.items
            item_json_result = {}
            for item in self.items:
                if item not in item_json_result:
                    item_json_result[item] = 1
                else:
                    item_json_result[item] += 1

        ## save the items json file 
        save_item_path = f'{self.args.items_folder}/{self.args.scene_name}.json'
        with open(save_item_path, "w") as file:
            json.dump(item_json_result, file)

        if self.args.items is None:
            self.items = list(set(list(item_json_result.keys())))


    def get_segmentation(self):
        
        ## load the items 
        if self.args.items is None:
            save_item_path = f'{self.args.items_folder}/{self.args.scene_name}.json'
            with open(save_item_path, "r") as file:
                item_json_result = json.load(file)
            self.items = list(item_json_result.keys())
        else:
            self.items = self.args.items
        
        print(self.items)

        ## get the segmentation results
        get_seg(self.scene_name, self.args.ref_image_folder, self.items, self.args.seg_folder)

        seg_flag = check_seg(self.args.seg_folder, self.scene_name, self.items)
        if not seg_flag:
            print("Segmentation check failed. Please check the segmentation results.")
            #exit(0)
        
        
        print(f"Segmentation data saved in {self.args.seg_folder}/{self.scene_name}")


    def get_object_description(self):

        get_obj_descrip(self.args, self.scene_name)

      


    def generate_objects(self):

        ## load the object description json file
        descrip_json_path = f'{self.args.descrip_folder}/{self.scene_name}.json'
        with open(descrip_json_path, "r") as file:
            object_descrip_dict = json.load(file)

        generate_image = False
        if generate_image:
            ## generate object images 
            generate_obj_ref_images(object_descrip_dict, self.args.ref_image_obj_candidate_folder, self.args.scene_name, self.args.ref_image_num, self.args.img_prompt_folder)
            
            ## select the best object image from the candidate images using vqascore
            #select_best_object_image(object_descrip_dict, self.args.ref_image_obj_candidate_folder, self.args.ref_image_obj_folder, self.scene_name, self.args.img_prompt_folder, self.args.vqas_env_path, self.args.vqa_score_folder)
    
        ## generate object 3d items
        select_objects = False 
        if select_objects:
            wanted_objects = ["cake1"]
            
            new_object_descrip_dict = {}
            for object_name in object_descrip_dict.keys():
                if object_name in wanted_objects:
                    new_object_descrip_dict[object_name] = object_descrip_dict[object_name]
            object_descrip_dict = new_object_descrip_dict
                
        generate_obj_items(object_descrip_dict, self.args.ref_image_obj_folder, self.args.raw_obj_folder, self.args.scene_name)

    def organize_objects(self):

        ## load the object description json file
        descrip_json_path = f'{self.args.descrip_folder}/{self.scene_name}.json'
        with open(descrip_json_path, "r") as file:
            object_descrip_dict = json.load(file)

        ## organize the object items into the folder 'data/clean_mesh/[scene_name]'
        for shape_name in object_descrip_dict.keys():
            organize_obj_items(self.args.raw_obj_folder, self.args.organized_obj_folder, self.args.scene_name, shape_name)
        
        ## origanize the duplicated items into the folder 'data/clean_mesh/[scene_name]'
        if self.args.dup:
            self.copy_duplicate_items()


    def copy_duplicate_items(self):
        duplicate_path = f'{self.args.duplicate_folder}/{self.scene_name}.json'
        duplicate_info = json.load(open(duplicate_path, 'r'))
        dublicate_item_name = list(duplicate_info.keys())[0]
        #print('dublicate_item_name:', dublicate_item_name)

        ## load item num 
        duplicate_item_num_path = f'{self.args.duplicate_folder}/{self.scene_name}_item_num.json'
        duplicate_item_num_info = json.load(open(duplicate_item_num_path, 'r'))
            
        output_scene_folder = f'{self.args.organized_obj_folder}/{self.args.scene_name}'
        input_scene_folder = f'{self.args.raw_obj_folder}/{self.args.scene_name}'

        ## duplicate the items with the same name for the number of times
        item_sons_dict = {}
        for item_name, item_num in duplicate_item_num_info.items():
            #print('item_name:', item_name)
            item_sons_dict[item_name] = []
            item_num = int(item_num)
            for i in range(item_num):    
                new_item_name = f'{item_name}{i}'
                item_sons_dict[item_name].append(new_item_name)


        for item_ori_item in item_sons_dict.keys():

            item_son_list = item_sons_dict[item_ori_item]
    
            for item_son_name in item_son_list:
                ## load the original mesh 
                texture_mesh_path = f'{input_scene_folder}/{item_ori_item}/{item_ori_item}_texture.obj'
                
                ## load the texture_mesh_path as txt list 
                with open(texture_mesh_path, 'r') as f:
                    texture_mesh = f.readlines()

                ## change the texture mesh 
                texture_mesh[1] = texture_mesh[1].replace('material', item_son_name)
                texture_mesh[2] = texture_mesh[2].replace('material_0', item_son_name)

                ## save the texture_mesh_path as txt list
                with open(f'{output_scene_folder}/{item_son_name}.obj', 'w') as f:
                    f.writelines(texture_mesh)

                ## change the mtl file
                mtl_path = f'{input_scene_folder}/{item_ori_item}/material.mtl'
                with open(mtl_path, 'r') as f:
                    mtl = f.readlines()
                mtl[2] = mtl[2].replace('material', item_son_name)
                mtl[7] = mtl[7].replace('material_0', item_son_name)

                ## save the texture_mesh_path as txt list
                with open(f'{output_scene_folder}/{item_son_name}.mtl', 'w') as f:
                    f.writelines(mtl)

                ## save the texture png
                os.system(f'cp {input_scene_folder}/{item_ori_item}/material_0.png {output_scene_folder}/{item_son_name}.png')
                

                ## remove the original mesh 
                os.system(f'rm {output_scene_folder}/{item_ori_item}.obj')
                os.system(f'rm {output_scene_folder}/{item_ori_item}.mtl')
                os.system(f'rm {output_scene_folder}/{item_ori_item}.png')
            
            








    def get_scene_layout(self):

        ## load the size ratio 
        size_path = f'{self.args.size_folder}/{self.scene_name}.json'
        with open(size_path, "r") as file:
            size_ratio = json.load(file)

        if self.args.dup:
            get_initial_layout_duplicate(self.args, self.args.organized_obj_folder, self.args.depth_folder, self.args.seg_folder, \
                                            self.args.scene_name, self.args.layout_folder, 100, size_ratio)
        else:
            get_initial_layout(self.args.organized_obj_folder, self.args.depth_folder, self.args.seg_folder, \
                            self.args.scene_name, self.args.layout_folder, self.args.layout_front_num,\
                            size_ratio)

        ## reset the y value of the objects
        ## move the object to above the ground and store the ground level value to the layout folder.
        reset_ground(self.args.layout_folder, self.args.depth_folder, self.args.scene_name, self.args.layout_folder)


    def get_low_poly(self):
        #if self.args.dup:
        #    get_low_poly_dup(self.args, self.args.scene_name, self.args.layout_folder, self.args.low_poly_folder, \
        #            self.args.low_poly_fnum, self.args.manifold_code_path)
        #else:
        get_low_poly_new(self.args.scene_name, self.args.layout_folder, self.args.low_poly_folder, \
                            self.args.low_poly_fnum, self.args.manifold_code_path)
     

    def get_contain_on_info(self):
   
        get_contain_on_info(self.args, self.scene_name)
   
    def get_size_info(self):
        
        get_size(self, self.args)
