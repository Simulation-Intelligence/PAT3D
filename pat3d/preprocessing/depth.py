import os
import subprocess
from dataclasses import replace
from PIL import Image
import numpy as np
import depth_pro
import torch
import pathlib
from plyfile import PlyData, PlyElement
import PIL.Image
from typing import Tuple, Any, Dict, Optional, Union


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _resolve_depth_pro_checkpoint() -> pathlib.Path:
    configured = os.environ.get("PAT3D_DEPTH_PRO_CHECKPOINT", "").strip()
    candidates = []
    if configured:
        candidates.append(pathlib.Path(configured).expanduser())
    candidates.extend(
        [
            REPO_ROOT / "extern" / "ml-depth-pro" / "checkpoints" / "depth_pro.pt",
            REPO_ROOT / "checkpoints" / "depth_pro.pt",
            pathlib.Path.cwd() / "checkpoints" / "depth_pro.pt",
        ]
    )
    for candidate in candidates:
        resolved = candidate if candidate.is_absolute() else REPO_ROOT / candidate
        if resolved.exists():
            return resolved
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "DepthPro checkpoint is missing. Set PAT3D_DEPTH_PRO_CHECKPOINT or place "
        f"depth_pro.pt under extern/ml-depth-pro/checkpoints/. Searched: {searched}"
    )


def _create_depth_pro_model_and_transforms():
    from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT

    checkpoint_path = _resolve_depth_pro_checkpoint()
    config = replace(
        DEFAULT_MONODEPTH_CONFIG_DICT,
        checkpoint_uri=str(checkpoint_path),
    )
    return depth_pro.create_model_and_transforms(config=config)



def resize_image(original_image: Union[PIL.Image.Image, np.ndarray], maximum_width: int, maximum_height: int):
    original_type = type(original_image)
    if isinstance(original_image, np.ndarray):
        image = PIL.Image.fromarray(original_image)

    width, height = image.size
    scale = 1.0
    if width > maximum_width or height > maximum_height:
        if width > height:
            scale = maximum_width / width
        else:
            scale = maximum_height / height
    else:
        return original_image
    new_width = int(width * scale)
    new_height = int(height * scale)
    image = image.resize((new_width, new_height), PIL.Image.NEAREST)
    if original_type == np.ndarray:
        image = np.array(image)
    return image

def save_point_cloud(pcd: torch.Tensor, rgb: np.ndarray, filename: pathlib.Path, binary: bool = True):
    """Save an RGB point cloud as a PLY file.
    :paras
        @pcd: Nx3 matrix, the XYZ coordinates
        @rgb: Nx3 matrix, the rgb colors for each 3D point
    """
    assert pcd.shape[0] == rgb.shape[0]

    if rgb is None:
        gray_concat = np.tile(np.array([128], dtype=np.uint8), (pcd.shape[0], 3))
        points_3d = np.hstack((pcd, gray_concat))
    else:
        points_3d = np.hstack((pcd, rgb))
    python_types = (float, float, float, int, int, int)
    npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    if binary is True:
        # Format into Numpy structured array
        vertices = []
        for row_idx in range(points_3d.shape[0]):
            cur_point = points_3d[row_idx]
            vertices.append(tuple(dtype(point) for dtype, point in zip(python_types, cur_point)))
        vertices_array = np.array(vertices, dtype=npy_types)
        el = PlyElement.describe(vertices_array, 'vertex')

        # write
        PlyData([el]).write(filename)
    else:
        x = np.squeeze(points_3d[:, 0])
        y = np.squeeze(points_3d[:, 1])
        z = np.squeeze(points_3d[:, 2])
        r = np.squeeze(points_3d[:, 3])
        g = np.squeeze(points_3d[:, 4])
        b = np.squeeze(points_3d[:, 5])

        ply_head = 'ply\n' \
                    'format ascii 1.0\n' \
                    'element vertex %d\n' \
                    'property float x\n' \
                    'property float y\n' \
                    'property float z\n' \
                    'property uchar red\n' \
                    'property uchar green\n' \
                    'property uchar blue\n' \
                    'end_header' % r.shape[0]
        # ---- Save ply data to disk
        np.savetxt(filename, np.column_stack([x, y, z, r, g, b]), fmt='%f %f %f %d %d %d', header=ply_head, comments='')


def get_pcd_base(H, W, u0, v0, fx, fy):
    
    x_row = np.arange(0, W)
    x = np.tile(x_row, (H, 1))
    x = x.astype(np.float32)
    u_m_u0 = x - u0

    y_col = np.arange(0, H) # y_col = np.arange(0, height)
    y = np.tile(y_col, (W, 1)).T
    y = y.astype(np.float32)
    v_m_v0 = y - v0

    x = u_m_u0 / fx
    y = v_m_v0 / fy
    z = np.ones_like(x)
    pw = np.stack([x, y, z], axis=2) # [h, w, c]
    return pw

def reconstruct_pcd(depth: torch.Tensor, fx: float, fy: float, u0: float, v0: float):
    H, W = depth.shape
    pcd_base = get_pcd_base(H, W, u0, v0, fx, fy)

    pcd = depth[:, :, None] * pcd_base
    return pcd


# TODO: calculate padding_transparent dynamially
def load_pil_image(
        image_path: pathlib.Path,
        padding_transparent: Union[Tuple[int, int, int], np.ndarray] = (127, 127, 127),
        maximum_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:

    with PIL.Image.open(image_path) as img:
        if maximum_size is not None:
            img = resize_image(img, *maximum_size)
            assert type(img) == PIL.Image.Image, f"Expected PIL.Image.Image, but got {type(img)}"

        if img.mode == 'RGBA':
            if isinstance(padding_transparent, np.ndarray):
                padding_transparent = tuple(padding_transparent)
            rgb_image = PIL.Image.new("RGB", img.size, padding_transparent)
            rgb_image.paste(img, mask=img.split()[3])
        else:
            rgb_image = img.convert('RGB')

    return np.array(rgb_image)

def get_depth(image_name, image_folder, output_dir):

    ## check the input folder to find the image with the name image_name
    input_img_path = os.path.join(image_folder, f'{image_name}.png')

    depth_output_dir_path = os.path.join(output_dir, image_name)
    if os.path.exists(depth_output_dir_path):
        rm_command = 'rm -r ' + depth_output_dir_path
        os.system(rm_command)
    os.makedirs(depth_output_dir_path)

    ## load model 
    model, transform = _create_depth_pro_model_and_transforms()
    model.eval()

    ## load and preprocess an image.
    image_ori, _, f_px = depth_pro.load_rgb(input_img_path)
    image = transform(image_ori)

    ## run inference.
    prediction = model.infer(image, f_px=f_px)
    depth = prediction["depth"]  # Depth in [m].

    ## visualize and save the depth
    depth_numpy = depth.squeeze().cpu().numpy()

    ## save the original depth array 
    np.save(os.path.join(depth_output_dir_path, f'{image_name}_depth.npy'), depth_numpy)


    depth_vis = (depth_numpy - np.min(depth_numpy)) / (np.max(depth_numpy) - np.min(depth_numpy))
    depth_img = Image.fromarray((depth_vis * 255).astype(np.uint8))
    depth_path = os.path.join(depth_output_dir_path, f'{image_name}_depth.png')
    depth_img.save(depth_path)

    ## get the predicted focal length
    focallength_px = prediction["focallength_px"]  # Focal length in pixels.

    #print('focallength_px:', focallength_px) ## 3105.2854
    #exit(0)

    ## project the depth to the point cloud
    R = 1.0
    F = focallength_px
    ORIGINAL_H, ORIGINAL_W = depth.shape[:2]

    #print('depth min:', np.min(depth_numpy), 'depth max:', np.max(depth_numpy))
    #exit(0)
    point_cloud = reconstruct_pcd(depth, F * R, F * (2 - R), ORIGINAL_W * 0.5, ORIGINAL_H * 0.5)

    point_cloud_path = os.path.join(depth_output_dir_path, f'{image_name}_point_cloud.npz')
    np.savez(point_cloud_path, point_cloud = point_cloud, focallength_px = focallength_px)

    point_cloud_vis_path = os.path.join(depth_output_dir_path, f'{image_name}_point_cloud.ply')
    save_point_cloud(point_cloud.reshape((-1, 3)), image_ori.reshape(-1, 3), point_cloud_vis_path)

    # Run the python command
    #subprocess.run(['python', 'sfg2/depth.py', '--input', input_img_path, '--depth_output', depth_output_dir_path])

