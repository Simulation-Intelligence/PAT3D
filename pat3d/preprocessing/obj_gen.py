import os
import gc
import torch
import sys
from pathlib import Path
from PIL import Image

from pat3d.runtime.hunyuan_env import ensure_hunyuan_runtime

REPO_ROOT = Path(__file__).resolve().parents[2]
HUNYUAN_ROOT = REPO_ROOT / "extern" / "Hunyuan3Dv2"
_HUNYUAN_ENV = ensure_hunyuan_runtime(
    repo_root=REPO_ROOT,
    hunyuan_root=HUNYUAN_ROOT,
)

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FaceReducer, FloaterRemover, DegenerateFaceRemover
from hy3dgen.text2image import HunyuanDiTPipeline
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from pat3d.preprocessing.img import generate_object_image


def _release_model_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, 'ipc_collect'):
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        if hasattr(libc, "malloc_trim"):
            libc.malloc_trim(0)
    except Exception:
        pass


def text_to_3d(prompt, image_obj_folder, save_obj_folder, scene_name, object_name):
    rembg = None
    t2i = None
    i23d = None
    pipeline = None
    image = None
    mesh = None
    try:
        rembg = BackgroundRemover()
        t2i = HunyuanDiTPipeline('Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled')
        i23d = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2mini',
                                                                subfolder='hunyuan3d-dit-v2-mini-turbo',
                                                                variant='fp16')

        #prompt = "Horizontal perspective and show the complete 3d object. " + prompt
        ## generate a random seed 
        random_seed = torch.randint(0, 1000000, (1,)).item()

        ## mkdir the folder if not exist
        image_obj_folder_scene = f'{image_obj_folder}/{scene_name}'
        if not os.path.exists(image_obj_folder_scene):
            os.makedirs(image_obj_folder_scene, exist_ok=True)
        
        item_image_path = f'{image_obj_folder}/{scene_name}/{object_name}.png'

        if not os.path.exists(item_image_path):
            image = t2i(prompt)
            image.save(item_image_path)
        #print(f"Object image not found: {item_image_path}")
        #exit(0)

        with Image.open(item_image_path) as loaded_image:
            image = loaded_image.copy()
        image = rembg(image)
        mesh = i23d(image, num_inference_steps=50, mc_algo='mc', seed = random_seed)[0]
        mesh = FloaterRemover()(mesh)
        mesh = DegenerateFaceRemover()(mesh)
        mesh = FaceReducer()(mesh)

        mesh_folder = f'{save_obj_folder}/{object_name}'
        ## remove 
        if os.path.exists(mesh_folder):
            os.system(f'rm -rf {mesh_folder}')
        os.makedirs(mesh_folder, exist_ok=True)

        mesh_path = f'{mesh_folder}/{object_name}_mesh.obj'
        mesh.export(mesh_path)

        t2i = None
        i23d = None
        rembg = None
        _release_model_memory()

        try:
            pipeline = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2')
            mesh = pipeline(mesh, image=image)
            mesh_texture_path = f'{mesh_folder}/{object_name}_texture.obj'
            mesh.export(mesh_texture_path)
        except Exception as exc:
            print(f'Warning: texture generation failed for {object_name}: {exc}')
    finally:
        pipeline = None
        mesh = None
        image = None
        i23d = None
        t2i = None
        rembg = None
        _release_model_memory()


def generate_obj_items(object_descrip_dict, ref_image_obj_folder, save_obj_folder, scene_name):

    ## create the folder if not exist
    scene_obj_folder = f'{save_obj_folder}/{scene_name}'
    if not(os.path.exists(scene_obj_folder)):
        os.system(f'mkdir -p {scene_obj_folder}')
    #    os.system(f'rm -rf {scene_obj_folder}')
    #os.makedirs(scene_obj_folder)

    ## generate the object items
    for object_name in object_descrip_dict.keys():
        object_descrip = object_descrip_dict[object_name]
        text_to_3d(object_descrip, ref_image_obj_folder, scene_obj_folder, scene_name, object_name)
    

def organize_obj_items(input_folder, output_folder, scene_name, shape_name):

    '--input_folder data/raw_obj --case_name guitar --output_dir data/clean_mesh/guitar_box'

    output_scene_folder = f'{output_folder}/{scene_name}'
    input_scene_folder = f'{input_folder}/{scene_name}'
    ## create the output folder
    if not os.path.exists(output_scene_folder):
        os.system(f'mkdir -p {output_scene_folder}')

    ## load the mesh 
    texture_mesh_path = f'{input_scene_folder}/{shape_name}/{shape_name}_texture.obj'

    ## load the texture_mesh_path as txt list 
    with open(texture_mesh_path, 'r') as f:
        texture_mesh = f.readlines()
    texture_mesh[1] = texture_mesh[1].replace('material', shape_name)
    texture_mesh[2] = texture_mesh[2].replace('material_0', shape_name)
    
    ## save the texture_mesh_path as txt list
    with open(f'{output_scene_folder}/{shape_name}.obj', 'w') as f:
        f.writelines(texture_mesh)

    ## change the mtl file
    mtl_path = f'{input_scene_folder}/{shape_name}/material.mtl'
    with open(mtl_path, 'r') as f:
        mtl = f.readlines()
    mtl[2] = mtl[2].replace('material', shape_name)
    mtl[7] = mtl[7].replace('material_0', shape_name)
   
    ## save the texture_mesh_path as txt list
    with open(f'{output_scene_folder}/{shape_name}.mtl', 'w') as f:
        f.writelines(mtl)

    ## save the texture png
    os.system(f'cp {input_scene_folder}/{shape_name}/material_0.png {output_scene_folder}/{shape_name}.png')
    

def generate_obj_ref_images(object_descrip_dict, ref_image_obj_folder, scene_name, ref_obj_image_num, img_utils_folder):

    ## mkdir the scene folder 
    scene_img_obj_folder = f'{ref_image_obj_folder}/{scene_name}'
    if os.path.exists(scene_img_obj_folder):
        os.system(f'rm -rf {scene_img_obj_folder}')
    os.makedirs(scene_img_obj_folder)
    print(f"Created folder: {scene_img_obj_folder}")

    ## get the object descriptions for each object and feed it to reve 
    api_key_path = f'{img_utils_folder}/reve_apikey.txt'

    for shape_name, full_text_prompt in object_descrip_dict.items():
        img_save_path_prefix = f'{scene_img_obj_folder}/{shape_name}'
        print(shape_name)
        additional_text_prompt = f"The image is taken in a horizontal perspective and show the complete 3d object. {full_text_prompt}"
        print(additional_text_prompt)
        continue
        if os.path.exists(img_save_path_prefix):
            os.system(f'rm -rf {img_save_path_prefix}')
        os.makedirs(img_save_path_prefix)
        generate_object_image(full_text_prompt, img_save_path_prefix, api_key_path, ref_obj_image_num, shape_name)



if __name__ == '__main__':
    prompt = "A realistic banana"
    save_obj_folder = "data/raw_obj"
    scene_name = 'inpaint_fail'
    object_name = "scissor"
    image_obj_folder = "data/ref_img_obj"
    text_to_3d(prompt, image_obj_folder, save_obj_folder, scene_name, object_name)
