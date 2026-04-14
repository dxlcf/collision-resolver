import sys
import numpy as np
import trimesh


def load_mesh(path: str) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh")

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"{path} 不是三角网格（Trimesh）")

    if mesh.faces is None or len(mesh.faces) == 0:
        raise ValueError(f"{path} 没有三角面，不能做 mesh 碰撞检测")

    return mesh


def make_translation(tx=0.0, ty=0.0, tz=0.0) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = [tx, ty, tz]
    return T


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python check_collision.py model_a.ply model_b.ply")
        sys.exit(1)

    path_a = sys.argv[1]
    path_b = sys.argv[2]

    mesh_a = load_mesh(path_a)
    mesh_b = load_mesh(path_b)

    # 可选：给第二个模型一个平移，方便你测试“撞上 / 没撞上”
    # 例如改成 [10, 0, 0] 就通常会分开很多
    T_b = make_translation(100.0, 0.0, 0.0)

    manager = trimesh.collision.CollisionManager()
    manager.add_object(mesh_a, mesh_a)

    collided, names, contacts = manager.in_collision_single(
        mesh_b,
        transform=T_b,
        return_names=True,
        return_data=True,
    )

    min_distance, nearest_name = manager.min_distance_single(
        mesh_b,
        transform=T_b,
        return_name=True,
        return_data=False,
    )

    print(f"碰撞: {collided}")
    print(f"发生碰撞的对象: {names}")
    print(f"最小距离: {min_distance}")

    if collided and contacts:
        c = contacts[0]
        print("第一组接触信息:")
        print("  接触点:", c.point)
        print("  接触法向:", c.normal)
        print("  穿透深度:", c.depth)