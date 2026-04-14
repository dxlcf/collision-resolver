import trimesh

def clean_mesh(input_path, output_path):
    mesh = trimesh.load(input_path, force='mesh')

    print(f"Before: watertight={mesh.is_watertight}, euler={mesh.euler_number}")

    # 1. 去重顶点
    mesh.merge_vertices()

    # 2. 删除退化面（新版写法）
    mesh.update_faces(mesh.nondegenerate_faces())

    # 3. 删除重复面
    mesh.update_faces(mesh.unique_faces())

    # 4. 删除孤立顶点
    mesh.remove_unreferenced_vertices()

    # 5. 修复法线
    mesh.fix_normals()

    # 6. 填洞（关键）
    trimesh.repair.fill_holes(mesh)

    # 7. 修复反转/体问题
    trimesh.repair.fix_inversion(mesh)

    # 8. 再清理一遍
    mesh.remove_unreferenced_vertices()

    print(f"After: watertight={mesh.is_watertight}, euler={mesh.euler_number}")

    mesh.export(output_path)


clean_mesh("data/mesh_c.ply", "data/mesh_c_clean.ply")