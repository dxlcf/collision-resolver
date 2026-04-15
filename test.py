import open3d as o3d
import numpy as np


def create_box_mesh(min_bound, max_bound):
    """
    创建一个轴对齐长方体三角网格
    """
    x0, y0, z0 = min_bound
    x1, y1, z1 = max_bound

    vertices = np.array([
        [x0, y0, z0],  # 0
        [x1, y0, z0],  # 1
        [x1, y1, z0],  # 2
        [x0, y1, z0],  # 3
        [x0, y0, z1],  # 4
        [x1, y0, z1],  # 5
        [x1, y1, z1],  # 6
        [x0, y1, z1],  # 7
    ], dtype=np.float64)

    triangles = np.array([
        [0, 1, 2], [0, 2, 3],  # bottom
        [4, 6, 5], [4, 7, 6],  # top
        [0, 4, 5], [0, 5, 1],  # front
        [1, 5, 6], [1, 6, 2],  # right
        [2, 6, 7], [2, 7, 3],  # back
        [3, 7, 4], [3, 4, 0],  # left
    ], dtype=np.int32)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    return mesh


def combine_meshes(mesh_list):
    """
    合并多个 TriangleMesh
    """
    vertices_all = []
    triangles_all = []
    v_offset = 0

    for mesh in mesh_list:
        v = np.asarray(mesh.vertices)
        t = np.asarray(mesh.triangles)

        vertices_all.append(v)
        triangles_all.append(t + v_offset)
        v_offset += len(v)

    out = o3d.geometry.TriangleMesh()
    out.vertices = o3d.utility.Vector3dVector(np.vstack(vertices_all))
    out.triangles = o3d.utility.Vector3iVector(np.vstack(triangles_all))
    out.compute_vertex_normals()
    return out


def build_open_box_and_small_cube():
    # -----------------------------
    # 1. 箱子参数
    # -----------------------------
    outer_size = np.array([1.0, 1.0, 0.9], dtype=np.float64)   # 外尺寸
    wall_thickness = 0.1
    bottom_thickness = 0.1

    ox, oy, oz = outer_size

    # 箱体内部范围
    inner_min = np.array([wall_thickness, wall_thickness, bottom_thickness], dtype=np.float64)
    inner_max = np.array([ox - wall_thickness, oy - wall_thickness, oz], dtype=np.float64)
    inner_size = inner_max - inner_min  # [0.8, 0.8, 0.8]

    # -----------------------------
    # 2. 构建开口箱子（5块板）
    # -----------------------------
    bottom = create_box_mesh(
        [0.0, 0.0, 0.0],
        [ox, oy, bottom_thickness]
    )

    left_wall = create_box_mesh(
        [0.0, 0.0, bottom_thickness],
        [wall_thickness, oy, oz]
    )

    right_wall = create_box_mesh(
        [ox - wall_thickness, 0.0, bottom_thickness],
        [ox, oy, oz]
    )

    front_wall = create_box_mesh(
        [wall_thickness, 0.0, bottom_thickness],
        [ox - wall_thickness, wall_thickness, oz]
    )

    back_wall = create_box_mesh(
        [wall_thickness, oy - wall_thickness, bottom_thickness],
        [ox - wall_thickness, oy, oz]
    )

    open_box = combine_meshes([bottom, left_wall, right_wall, front_wall, back_wall])

    # -----------------------------
    # 3. 构建更小的正方体
    # -----------------------------
    cube_size = 0.5

    if np.any(inner_size < cube_size):
        raise ValueError("正方体太大，放不进箱子内部")

    # 先创建单位立方体，再缩放到 cube_size
    cube = o3d.geometry.TriangleMesh.create_box(
        width=cube_size,
        height=cube_size,
        depth=cube_size
    )
    cube.compute_vertex_normals()

    # -----------------------------
    # 4. 调整正方体位姿
    #    目标：放在箱子内部中央，并落在箱底上
    # -----------------------------
    target_x = inner_min[0] + (inner_size[0] - cube_size) / 2.0
    target_y = inner_min[1] + (inner_size[1] - cube_size) / 2.0
    target_z = bottom_thickness

    cube.translate([target_x, target_y, target_z])

    # -----------------------------
    # 5. 上色与导出
    # -----------------------------
    open_box.paint_uniform_color([0.7, 0.7, 0.7])
    cube.paint_uniform_color([0.9, 0.3, 0.3])

    o3d.io.write_triangle_mesh("open_box.ply", open_box)
    o3d.io.write_triangle_mesh("inner_cube.ply", cube)

    return open_box, cube


if __name__ == "__main__":
    open_box, cube = build_open_box_and_small_cube()

    print("已导出:")
    print("  - open_box.ply")
    print("  - inner_cube.ply")

    o3d.visualization.draw_geometries(
        [open_box, cube],
        window_name="Open Box with Small Cube",
        mesh_show_back_face=True
    )