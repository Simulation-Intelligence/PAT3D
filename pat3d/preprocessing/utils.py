import glob
import os

import matplotlib.image as mpimg
import numpy as np

def load_image(save_img_path):
    """
    Load an image from a specified path using glob to match the file pattern.
    
    Args:
        save_img_path (str): The path pattern to the image file.
        
    Returns:
        None: The image is loaded into self.scene_image.
    """
    # Use glob to find files matching the pattern
    matching_files = glob.glob(save_img_path)
    if matching_files:
        # Take the first matching file
        image_path = matching_files[0]
        scene_image = mpimg.imread(image_path)
    else:
        print(f"No image found for pattern: {save_img_path}")
        scene_image = None
    return scene_image


def draw_colored_points_to_obj(save_path, pc):
    points = np.asarray(pc)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected point cloud with shape (N, >=3), got {points.shape}")

    output_dir = os.path.dirname(save_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    xyz = points[:, :3]
    has_color = points.shape[1] >= 6
    rgb = None
    if has_color:
        rgb = np.asarray(points[:, 3:6])
        if np.issubdtype(rgb.dtype, np.floating):
            if rgb.max(initial=0.0) <= 1.0:
                rgb = np.clip(rgb * 255.0, 0, 255)
            rgb = np.rint(rgb)
        rgb = np.clip(rgb, 0, 255).astype(int)

    with open(save_path, "w") as f:
        if has_color:
            for (x, y, z), (r, g, b) in zip(xyz, rgb):
                f.write(f"v {x:.8f} {y:.8f} {z:.8f} {r:d} {g:d} {b:d}\n")
        else:
            for x, y, z in xyz:
                f.write(f"v {x:.8f} {y:.8f} {z:.8f}\n")
