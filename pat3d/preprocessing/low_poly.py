import os
import sys
import json 
import subprocess
import tempfile
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..")) 
sys.path.append(project_root)

import trimesh
import argparse


from pat3d.preprocessing.mesh_utils.remesh_low_poly import remesh_low_polygon
from pat3d.preprocessing.mesh_utils.poisson_recon import get_recon_mesh, get_max_connected_region


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case_name', type=str, default=None)
    parser.add_argument('--input_root', type=str, default='data/layout')
    parser.add_argument('--output_root', type=str, default='data/low_poly')
    parser.add_argument('--poisson_samples', type=int, default=300000,
                        help = 'The number of the samples for poisson reconstruction.')
    parser.add_argument('--target_face_num', type=int, default=2000,
                        help = 'The target face number for the final low polygon.')
    return parser.parse_args()


## load objects 
## input: the path for the input folder 
## output: a dictionary with key as the file name(with .obj) and value as the trimesh mesh object
def load_obj_from_root(root_path):

    obj_dict = {}
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith('.obj'):
                obj_path = os.path.join(root, file)
                obj = trimesh.load(obj_path)
                obj_dict[file] = obj

    return obj_dict

def load_obj_path_from_root(root_path):

    obj_path_dict = {}
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith('.obj'):
                obj_path_dict[file] = f'{root_path}/{file}'

    return obj_path_dict


def resolve_float_tetwild_path():
    for env_var_name in ('PAT3D_FTETWILD_BIN', 'FTETWILD_BIN'):
        env_path = os.environ.get(env_var_name)
        if env_path and os.path.exists(env_path):
            return env_path

    repo_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    candidate_paths = [
        os.path.join(repo_root, 'extern', 'fTetWild', 'build', 'FloatTetwild_bin'),
        os.path.join(repo_root, 'extern', 'fTetWild', 'build-conda', 'FloatTetwild_bin'),
    ]
    for candidate_path in candidate_paths:
        if os.path.exists(candidate_path):
            return candidate_path

    raise FileNotFoundError(
        'FloatTetwild_bin not found. Set PAT3D_FTETWILD_BIN/FTETWILD_BIN or build extern/fTetWild/build/FloatTetwild_bin.'
    )


def resolve_target_face_num(
    requested_target_face_num,
    area: float,
    *,
    min_face_num: int,
    max_face_num: int,
    per_area_face_num: int,
) -> int:
    if requested_target_face_num is not None and int(requested_target_face_num) > 0:
        return max(min(int(requested_target_face_num), max_face_num), min_face_num)

    computed_target_face_num = int(area * per_area_face_num)
    computed_target_face_num = min(computed_target_face_num, max_face_num)
    computed_target_face_num = max(computed_target_face_num, min_face_num)
    return computed_target_face_num


def resolve_max_tetwild_input_faces(default: int = 50000) -> int:
    raw_value = os.environ.get("PAT3D_TETWILD_MAX_INPUT_FACES", "").strip()
    if not raw_value:
        return default
    try:
        parsed_value = int(raw_value)
    except ValueError:
        return default
    return max(parsed_value, 0)


def _load_mesh_for_tetwild(input_obj_path: str) -> trimesh.Trimesh:
    loaded = trimesh.load(input_obj_path, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.dump(concatenate=True)
    return loaded.copy()


def _prepare_input_mesh_for_tetwild(
    input_obj_path: str,
    *,
    max_input_faces: int,
) -> tuple[str, tempfile.TemporaryDirectory[str] | None]:
    if max_input_faces <= 0:
        return input_obj_path, None

    mesh = _load_mesh_for_tetwild(input_obj_path)
    if len(mesh.faces) <= max_input_faces:
        return input_obj_path, None

    temp_dir = tempfile.TemporaryDirectory(prefix="pat3d-tetwild-input-")
    temp_path = os.path.join(temp_dir.name, os.path.basename(input_obj_path))
    try:
        simplified = mesh.simplify_quadric_decimation(face_count=max_input_faces, aggression=8)
        if simplified is None or len(simplified.faces) == 0:
            raise RuntimeError("quadric simplification produced an empty mesh")
        simplified.export(temp_path)
        return temp_path, temp_dir
    except Exception:
        try:
            mesh.export(temp_path)
            remesh_low_polygon(temp_path, temp_path, max_input_faces)
            simplified = _load_mesh_for_tetwild(temp_path)
            if len(simplified.faces) == 0:
                raise RuntimeError("pymeshlab simplification produced an empty mesh")
            return temp_path, temp_dir
        except Exception:
            temp_dir.cleanup()
            return input_obj_path, None


def _run_float_tetwild(input_obj_path: str, output_prefix: str) -> str:
    float_tetwild_path = resolve_float_tetwild_path()
    prepared_input_path, temp_dir = _prepare_input_mesh_for_tetwild(
        input_obj_path,
        max_input_faces=resolve_max_tetwild_input_faces(),
    )
    try:
        command = [
            float_tetwild_path,
            "-i",
            prepared_input_path,
            "-o",
            output_prefix,
            "-l", 
            "0.1",
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        detail = "\n".join(
            chunk.strip()
            for chunk in (completed.stdout or "", completed.stderr or "")
            if chunk and chunk.strip()
        ).strip()
        surface_mesh_path = f"{output_prefix}__sf.obj"
        if completed.returncode != 0:
            raise RuntimeError(
                f"FloatTetWild failed with exit code {completed.returncode}: "
                f"{detail or 'no process output was captured'}"
            )
        if not os.path.exists(surface_mesh_path):
            raise FileNotFoundError(
                f"FloatTetWild did not produce the expected surface mesh '{surface_mesh_path}'. "
                f"{detail or 'no process output was captured'}"
            )
        return surface_mesh_path
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def get_low_poly_new(scene_name, layout_folder, low_poly_folder, target_face_num, manifold_code_path):

    ## load the objects
    case_path = os.path.join(layout_folder, scene_name)
    ori_obj_path_dict = load_obj_path_from_root(case_path)

    max_face_num = 4000 
    min_face_num = 500
    per_area_face_num = 6000
    
    for obj_name, obj_path in ori_obj_path_dict.items():
        output_prefix = os.path.join(low_poly_folder, scene_name)
        os.makedirs(output_prefix, exist_ok=True)
        output_path = os.path.join(output_prefix, obj_name)

        surface_mesh_path = _run_float_tetwild(obj_path, output_path[:-4])
        
        repair_mesh = trimesh.load(surface_mesh_path)

        ## get the max connected region
        repair_mesh = get_max_connected_region(repair_mesh)
        if not(repair_mesh.is_watertight):
            print(f'{obj_name} is not watertight')
            # exit(0)

        repair_mesh.export(output_path)

        ## get the target face number
        target_face_num = resolve_target_face_num(
            target_face_num,
            repair_mesh.area,
            min_face_num=min_face_num,
            max_face_num=max_face_num,
            per_area_face_num=per_area_face_num,
        )

        ## remesh the geometry
        remesh_low_polygon(output_path, output_path, target_face_num)

        low_mesh = trimesh.load(output_path)
        low_mesh.export(output_path)
        low_mesh_watertight_flag = low_mesh.is_watertight

        if low_mesh_watertight_flag:
            print(f'{obj_name} is watertight')
        else:
            print(f'{obj_name} is not watertight')
            # exit(0)


    # --- Remove redundant files ---
    # List of expected output files (full paths)
    expected_files = set([
        os.path.join(low_poly_folder, scene_name, obj_name)
        for obj_name in ori_obj_path_dict.keys()
    ])

    # rename the watertight mesh after tetwild 
    output_dir = os.path.join(low_poly_folder, scene_name)
    for fname in os.listdir(output_dir):
        fpath = os.path.join(output_dir, fname)
        # Remove file if not in expected_files
        if os.path.isfile(fpath) and fpath not in expected_files:
            if fname.lower().endswith('sf.obj'):
                new_name = fname.replace('__sf.obj', '_beforereduction.obj')
                new_path = os.path.join(output_dir, new_name)
                os.rename(fpath, new_path)
    
    ## remove all the files that do not end in obj 
    remove_files_list = []
    for fname in os.listdir(output_dir):
        fpath = os.path.join(output_dir, fname)
        if not fname.lower().endswith('obj'):
            remove_files_list.append(fpath)
            os.remove(fpath)

    #print(f'remove_files_list: {remove_files_list}')
    #exit(0)



def get_low_poly_from_name_list(scene_name, layout_folder, low_poly_folder, target_face_num, name_list):

    ## load the objects
    case_path = os.path.join(layout_folder, scene_name)
    ori_obj_path_dict = {}

    for obj_name in name_list:
        obj_path = os.path.join(case_path, f'{obj_name}.obj')

        if not os.path.exists(obj_path):
            print(f'{obj_name} does not exist')
            continue
        ori_obj_path_dict[obj_name] = obj_path

    #print(f'ori_obj_path_dict: {ori_obj_path_dict}')
    #exit(0)

    max_face_num = 4000 
    min_face_num = 500
    per_area_face_num = 6000
    
    for obj_name, obj_path in ori_obj_path_dict.items():
        output_prefix = os.path.join(low_poly_folder, scene_name)
        #os.makedirs(output_prefix, exist_ok=True)
        output_path = os.path.join(output_prefix, f'{obj_name}.obj')

        surface_mesh_path = _run_float_tetwild(obj_path, output_path[:-4])
        
        repair_mesh = trimesh.load(surface_mesh_path)

        ## get the max connected region
        repair_mesh = get_max_connected_region(repair_mesh)
        if not(repair_mesh.is_watertight):
            print(f'{obj_name} is not watertight')
            # exit(0)

        repair_mesh.export(output_path)

        ## get the target face number
        target_face_num = resolve_target_face_num(
            target_face_num,
            repair_mesh.area,
            min_face_num=min_face_num,
            max_face_num=max_face_num,
            per_area_face_num=per_area_face_num,
        )

        ## remesh the geometry
        remesh_low_polygon(output_path, output_path, target_face_num)

        low_mesh = trimesh.load(output_path)
        low_mesh.export(output_path)
        low_mesh_watertight_flag = low_mesh.is_watertight

        if low_mesh_watertight_flag:
            print(f'{obj_name} is watertight')
        else:
            print(f'{obj_name} is not watertight')
            # exit(0)


    # --- Remove redundant files ---
    # List of expected output files (full paths)
    expected_files = set([
        os.path.join(low_poly_folder, scene_name, obj_name)
        for obj_name in ori_obj_path_dict.keys()
    ])
    
    # List all files in the output directory
    output_dir = os.path.join(low_poly_folder, scene_name)
    for fname in os.listdir(output_dir):
        fpath = os.path.join(output_dir, fname)
        # Remove file if not in expected_files
        if os.path.isfile(fpath) and fpath not in expected_files:
            if fname.lower().endswith('sf.obj'):
                new_name = fname.replace('__sf.obj', '_beforereduction.obj')
                new_path = os.path.join(output_dir, new_name)
                os.rename(fpath, new_path)
                print(f"Renamed {fpath} to {new_path}")
                #os.remove(new_path)






if __name__ == '__main__':

    scene_name = 'largestudy'
    layout_folder = f'data/layout'
    low_poly_folder = f'data/low_poly'
    target_face_num = 2000
    name_list =  ["book6"]
    get_low_poly_from_name_list(scene_name, layout_folder, low_poly_folder, target_face_num, name_list)
    
