import os
import numpy as np
import trimesh
from scipy.spatial import ConvexHull
import trimesh 
import igl
import open3d as o3d
import pymeshfix
from scipy.spatial import KDTree 



def sample_points_on_mesh(cur_mesh, sample_surf_num):
    cur_surface_points, fids = trimesh.sample.sample_surface(cur_mesh, sample_surf_num)
    cur_surface_normals = cur_mesh.face_normals[fids]
    return cur_surface_points, cur_surface_normals

def sample_point_cloud(mesh_piece, sample_num):

    ## sample the point cloud with normals
    sampled_point_cloud, sampled_normals = sample_points_on_mesh(mesh_piece, sample_num) 

    ## compute the normals according to the point cloud
    #pcd = o3d.geometry.PointCloud()
    #pcd.points = o3d.utility.Vector3dVector(sampled_point_cloud)
    #pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    ## create o3d point cloud data
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(sampled_point_cloud)
    point_cloud.normals = o3d.utility.Vector3dVector(sampled_normals)

    #o3d.io.write_point_cloud(point_cloud_path, point_cloud)
    return point_cloud
    #return pcd

def try_pymeshfix(mesh):

    tin = pymeshfix.PyTMesh()
    #tin.clean(max_iters=10, inner_loops=3)
    meshfix = pymeshfix.MeshFix(mesh.vertices, mesh.faces)
    try:
        meshfix.repair(remove_smallest_components=False)
    except:
        print("repair failed")
    
    tin.load_array(meshfix.v, meshfix.f)
    tin.fill_small_boundaries()
    tin.clean(max_iters=10, inner_loops=3)
    v, f = tin.return_arrays()
    new_mesh = trimesh.Trimesh(v, f)
    
    return new_mesh


def poisson_reconstruction(point_cloud, repair_mesh_path, depth):
    
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=depth)

    o3d.io.write_triangle_mesh(repair_mesh_path, mesh)

## poisson reconstruction to get a simple mesh
## input: obj_mesh: the trimesh mesh object
##        poisson_samples: the number of samples for poisson reconstruction
## output: the poisson reconstruction mesh
def get_recon_mesh(obj_mesh, poisson_samples, repair_mesh_path, depth = 6):

    ## sample the point cloud
    point_cloud_piece = sample_point_cloud(obj_mesh, poisson_samples)

    ## poisson reconstruction and export the mesh
    poisson_reconstruction(point_cloud_piece, repair_mesh_path, depth = depth)


def get_max_connected_region(mesh):

    components = mesh.split(only_watertight=False) 
    volumes = [component.volume for component in components]  

    max_component = components[np.argmax(volumes)]
    ## return the max connected region
    return max_component


def try_pymeshfix(mesh):

    tin = pymeshfix.PyTMesh()
    #tin.clean(max_iters=10, inner_loops=3)
    meshfix = pymeshfix.MeshFix(mesh.vertices, mesh.faces)
    try:
        meshfix.repair(remove_smallest_components=False)
    except:
        print("repair failed")
    
    tin.load_array(meshfix.v, meshfix.f)
    tin.fill_small_boundaries()
    tin.clean(max_iters=10, inner_loops=3)
    v, f = tin.return_arrays()
    new_mesh = trimesh.Trimesh(v, f)
    
    return new_mesh