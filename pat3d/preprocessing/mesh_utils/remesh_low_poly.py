import pymeshlab


def remesh_low_polygon(input_mesh_path, output_mesh_path, target_face_num):
    ms = pymeshlab.MeshSet()

    ms.load_new_mesh(input_mesh_path)

    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum = target_face_num,  # 目标面片数量
        preservenormal=True,  # 是否保持法线
        preserveboundary=True,  # 是否保持边界
        preservetopology=True,  # 是否保持拓扑结构
        qualitythr=1.0,  # 简化质量阈值
        optimalplacement=True,  # 是否启用顶点的最佳位置
        planarquadric=True,  # 是否使用平面二次误差
        autoclean=True  # 是否自动清理网格
    )

    ms.save_current_mesh(output_mesh_path)
  