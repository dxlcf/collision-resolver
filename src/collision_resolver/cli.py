from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import open3d as o3d
from loguru import logger

from collision_resolver.formula_collision import (
    JointOptimizationReport,
    SymmetricLossReport,
    evaluate_symmetric_collision_loss,
    identity_transform,
    optimize_joint_transforms,
    validate_transform_matrix,
)
from collision_resolver.preprocess_cache import DEFAULT_CACHE_DIR, preprocess_mesh_with_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute symmetric SDF collision loss with offline SDF and offline surface point clouds."
        ),
    )
    parser.add_argument("mesh_a", help="Path of mesh A.")
    parser.add_argument("mesh_b", help="Path of mesh B.")
    parser.add_argument(
        "--transform-a",
        nargs=16,
        type=float,
        default=None,
        metavar="Aij",
        help=(
            "Row-major 4x4 transform matrix of mesh A (16 values). "
            "If omitted, identity transform is used."
        ),
    )
    parser.add_argument(
        "--transform-b",
        nargs=16,
        type=float,
        default=None,
        metavar="Bij",
        help=(
            "Row-major 4x4 transform matrix of mesh B (16 values). "
            "If omitted, identity transform is used."
        ),
    )
    parser.add_argument(
        "--transform-a-file",
        type=str,
        default=None,
        help="Optional path to a 4x4 transform file for mesh A (.npy or text).",
    )
    parser.add_argument(
        "--transform-b-file",
        type=str,
        default=None,
        help="Optional path to a 4x4 transform file for mesh B (.npy or text).",
    )
    parser.add_argument(
        "--sdf-cache-dir",
        type=str,
        default=str(DEFAULT_CACHE_DIR),
        help="Directory for preprocessed watertight mesh, SDF cache, and offline surface points.",
    )
    parser.add_argument(
        "--rebuild-preprocess-cache",
        action="store_true",
        help="Force rebuilding preprocess cache even if cache already exists.",
    )
    parser.add_argument(
        "--surface-point-count",
        type=int,
        default=20_000,
        help="Offline surface point count stored in preprocess cache for each mesh.",
    )
    parser.add_argument(
        "--voxel-size-ratio",
        type=float,
        default=1e-2,
        help="SDF voxel size ratio against mesh bbox diagonal.",
    )
    parser.add_argument(
        "--padding-ratio",
        type=float,
        default=5e-2,
        help="SDF bbox padding ratio against mesh bbox diagonal.",
    )
    parser.add_argument(
        "--max-grid-dim",
        type=int,
        default=160,
        help="Upper bound of SDF grid resolution per axis.",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Enable joint optimization for A/B transforms.",
    )
    parser.add_argument(
        "--max-opt-iters",
        type=int,
        default=30,
        help="Maximum number of iterations for joint optimization.",
    )
    parser.add_argument(
        "--opt-grad-eps",
        type=float,
        default=1e-4,
        help="Central-difference epsilon for optimization gradient.",
    )
    parser.add_argument(
        "--opt-init-step",
        type=float,
        default=1.0,
        help="Initial line-search step size for optimization.",
    )
    parser.add_argument(
        "--opt-backtrack-factor",
        type=float,
        default=0.5,
        help="Backtracking factor in optimization line search.",
    )
    parser.add_argument(
        "--opt-armijo-c",
        type=float,
        default=1e-4,
        help="Armijo condition coefficient in optimization line search.",
    )
    parser.add_argument(
        "--opt-grad-tol",
        type=float,
        default=1e-6,
        help="Gradient norm tolerance for optimization convergence.",
    )
    parser.add_argument(
        "--opt-loss-tol",
        type=float,
        default=1e-10,
        help="Loss tolerance for optimization convergence.",
    )
    parser.add_argument(
        "--opt-min-step",
        type=float,
        default=1e-8,
        help="Minimum line-search step size before optimization is considered failed.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize the current/final transformed meshes and penetration points.",
    )
    parser.add_argument(
        "--optimize-visualize",
        action="store_true",
        help="When optimizing, visualize both before and after optimization states.",
    )
    return parser.parse_args()


def format_vec(vec: np.ndarray | None) -> str:
    if vec is None:
        return "N/A"
    return f"[{vec[0]:.6f}, {vec[1]:.6f}, {vec[2]:.6f}]"


def _matrix_from_flat(values: list[float], label: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64).reshape(4, 4)
    return validate_transform_matrix(matrix, label)


def _matrix_from_file(path: str | Path, label: str) -> np.ndarray:
    path_obj = Path(path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"{label} file does not exist: {path_obj}")

    if path_obj.suffix.lower() == ".npy":
        loaded = np.load(path_obj)
    else:
        loaded = np.loadtxt(path_obj, dtype=np.float64)

    matrix = np.asarray(loaded, dtype=np.float64)
    if matrix.shape != (4, 4):
        if matrix.size != 16:
            raise ValueError(f"{label} from file must contain exactly 16 values.")
        matrix = matrix.reshape(4, 4)

    return validate_transform_matrix(matrix, label)


def resolve_transform(
    *,
    flat_values: list[float] | None,
    file_path: str | None,
    label: str,
) -> np.ndarray:
    if flat_values is not None and file_path is not None:
        raise ValueError(f"{label} supports either inline values or file input, not both.")
    if flat_values is not None:
        return _matrix_from_flat(flat_values, label)
    if file_path is not None:
        return _matrix_from_file(file_path, label)
    return identity_transform()


def _print_matrix(name: str, matrix: np.ndarray) -> None:
    print(f"{name}:")
    for row in matrix:
        print(f"  [{row[0]: .6f}, {row[1]: .6f}, {row[2]: .6f}, {row[3]: .6f}]")


def print_report(
    *,
    title: str,
    args: argparse.Namespace,
    transform_a: np.ndarray,
    transform_b: np.ndarray,
    report: SymmetricLossReport,
) -> None:
    print("=" * 72)
    print(f"{title:^72}")
    print("=" * 72)
    print(f"Mesh A: {Path(args.mesh_a)}")
    print(f"Mesh B: {Path(args.mesh_b)}")
    print(f"Surface samples (offline): {args.surface_point_count}")
    _print_matrix("Transform A", transform_a)
    _print_matrix("Transform B", transform_b)

    print("\n[B -> A]")
    print(f"Samples: {report.result_b_to_a.sample_count}")
    print(f"Penetrating samples: {report.result_b_to_a.penetrating_count}")
    print(f"Max penetration depth: {report.result_b_to_a.max_penetration_depth:.6f}")
    print(f"Loss_B_to_A: {report.result_b_to_a.loss:.10f}")
    print(f"BBox min: {format_vec(report.result_b_to_a.bbox_min)}")
    print(f"BBox max: {format_vec(report.result_b_to_a.bbox_max)}")

    print("\n[A -> B]")
    print(f"Samples: {report.result_a_to_b.sample_count}")
    print(f"Penetrating samples: {report.result_a_to_b.penetrating_count}")
    print(f"Max penetration depth: {report.result_a_to_b.max_penetration_depth:.6f}")
    print(f"Loss_A_to_B: {report.result_a_to_b.loss:.10f}")
    print(f"BBox min: {format_vec(report.result_a_to_b.bbox_min)}")
    print(f"BBox max: {format_vec(report.result_a_to_b.bbox_max)}")

    print("\n[Overall]")
    print(f"Collision: {'YES' if report.overall_collision else 'NO'}")
    print(f"Max penetration depth: {report.overall_depth:.6f}")
    print(f"Loss_total: {report.total_loss:.10f}")
    print(f"BBox min: {format_vec(report.overall_bbox_min)}")
    print(f"BBox max: {format_vec(report.overall_bbox_max)}")
    print("=" * 72)


def print_optimization_summary(report: JointOptimizationReport) -> None:
    print("\n[Joint Optimization]")
    print(f"Success: {'YES' if report.success else 'NO'}")
    print(f"Stop reason: {report.stop_reason}")
    print(f"Iterations: {report.iterations}")
    print(f"Initial loss: {report.initial_loss:.10f}")
    print(f"Final loss: {report.final_loss:.10f}")
    print(f"Loss history length: {len(report.loss_history)}")


def o3d_mesh_to_pv(mesh: o3d.geometry.TriangleMesh, pv_module):
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    faces = np.hstack(
        [np.full((len(triangles), 1), 3, dtype=np.int64), triangles.astype(np.int64)],
    )
    return pv_module.PolyData(vertices, faces)


def transformed_mesh(
    mesh: o3d.geometry.TriangleMesh,
    transform: np.ndarray,
) -> o3d.geometry.TriangleMesh:
    mesh_copy = copy.deepcopy(mesh)
    mesh_copy.transform(np.asarray(transform, dtype=np.float64))
    return mesh_copy


def visualize_report(
    *,
    mesh_a_local: o3d.geometry.TriangleMesh,
    mesh_b_local: o3d.geometry.TriangleMesh,
    transform_a: np.ndarray,
    transform_b: np.ndarray,
    report: SymmetricLossReport,
    title: str,
) -> None:
    try:
        import pyvista as pv
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Visualization requires pyvista. Install optional dependency first.",
        ) from exc

    mesh_a_world = transformed_mesh(mesh_a_local, transform_a)
    mesh_b_world = transformed_mesh(mesh_b_local, transform_b)

    plotter = pv.Plotter(title=title)
    plotter.add_mesh(o3d_mesh_to_pv(mesh_a_world, pv), color="cyan", opacity=0.22, label="Mesh A")
    plotter.add_mesh(o3d_mesh_to_pv(mesh_b_world, pv), color="magenta", opacity=0.22, label="Mesh B")

    points_b_to_a = report.result_b_to_a.penetration_points_world
    if len(points_b_to_a) > 0:
        plotter.add_points(
            points_b_to_a,
            color="yellow",
            point_size=4,
            render_points_as_spheres=True,
            label="B->A Penetration",
        )

    points_a_to_b = report.result_a_to_b.penetration_points_world
    if len(points_a_to_b) > 0:
        plotter.add_points(
            points_a_to_b,
            color="orange",
            point_size=4,
            render_points_as_spheres=True,
            label="A->B Penetration",
        )

    if report.overall_bbox_min is not None and report.overall_bbox_max is not None:
        bbox_mesh = pv.Box(
            [
                report.overall_bbox_min[0],
                report.overall_bbox_max[0],
                report.overall_bbox_min[1],
                report.overall_bbox_max[1],
                report.overall_bbox_min[2],
                report.overall_bbox_max[2],
            ],
        )
        plotter.add_mesh(
            bbox_mesh.scale(1.01),
            color="lime",
            style="wireframe",
            line_width=6,
            label="Collision BBox",
        )

    plotter.add_legend()
    plotter.show()


def run(args: argparse.Namespace) -> None:
    if args.optimize_visualize:
        args.optimize = True

    transform_a = resolve_transform(
        flat_values=args.transform_a,
        file_path=args.transform_a_file,
        label="transform_a",
    )
    transform_b = resolve_transform(
        flat_values=args.transform_b,
        file_path=args.transform_b_file,
        label="transform_b",
    )

    cache_root = Path(args.sdf_cache_dir)
    mesh_a_asset = preprocess_mesh_with_cache(
        args.mesh_a,
        cache_root=cache_root,
        force_rebuild=args.rebuild_preprocess_cache,
        voxel_size_ratio=args.voxel_size_ratio,
        padding_ratio=args.padding_ratio,
        max_grid_dim=args.max_grid_dim,
        surface_point_count=args.surface_point_count,
    )
    mesh_b_asset = preprocess_mesh_with_cache(
        args.mesh_b,
        cache_root=cache_root,
        force_rebuild=args.rebuild_preprocess_cache,
        voxel_size_ratio=args.voxel_size_ratio,
        padding_ratio=args.padding_ratio,
        max_grid_dim=args.max_grid_dim,
        surface_point_count=args.surface_point_count,
    )
    logger.info(
        "Preprocess cache: mesh_a={}, mesh_b={}",
        "hit" if mesh_a_asset.cache_hit else "build",
        "hit" if mesh_b_asset.cache_hit else "build",
    )

    initial_report = evaluate_symmetric_collision_loss(
        asset_a=mesh_a_asset,
        asset_b=mesh_b_asset,
        transform_a=transform_a,
        transform_b=transform_b,
    )

    optimized_transform_a = transform_a
    optimized_transform_b = transform_b
    final_report = initial_report
    optimization_report: JointOptimizationReport | None = None

    if args.optimize:
        optimization_report = optimize_joint_transforms(
            asset_a=mesh_a_asset,
            asset_b=mesh_b_asset,
            transform_a_init=transform_a,
            transform_b_init=transform_b,
            max_iters=args.max_opt_iters,
            grad_eps=args.opt_grad_eps,
            init_step=args.opt_init_step,
            backtrack_factor=args.opt_backtrack_factor,
            armijo_c=args.opt_armijo_c,
            grad_tol=args.opt_grad_tol,
            loss_tol=args.opt_loss_tol,
            min_step=args.opt_min_step,
        )
        optimized_transform_a = optimization_report.transform_a_optimized
        optimized_transform_b = optimization_report.transform_b_optimized
        final_report = optimization_report.final_report

    if args.optimize:
        print_report(
            title="SYMMETRIC SDF LOSS (BEFORE OPT)",
            args=args,
            transform_a=transform_a,
            transform_b=transform_b,
            report=initial_report,
        )
        if optimization_report is not None:
            print_optimization_summary(optimization_report)
        print_report(
            title="SYMMETRIC SDF LOSS (AFTER OPT)",
            args=args,
            transform_a=optimized_transform_a,
            transform_b=optimized_transform_b,
            report=final_report,
        )
    else:
        print_report(
            title="SYMMETRIC SDF COLLISION LOSS",
            args=args,
            transform_a=transform_a,
            transform_b=transform_b,
            report=initial_report,
        )

    if args.optimize_visualize:
        visualize_report(
            mesh_a_local=mesh_a_asset.mesh,
            mesh_b_local=mesh_b_asset.mesh,
            transform_a=transform_a,
            transform_b=transform_b,
            report=initial_report,
            title="Symmetric SDF Visualization - Before Optimization",
        )
        visualize_report(
            mesh_a_local=mesh_a_asset.mesh,
            mesh_b_local=mesh_b_asset.mesh,
            transform_a=optimized_transform_a,
            transform_b=optimized_transform_b,
            report=final_report,
            title="Symmetric SDF Visualization - After Optimization",
        )
    elif args.visualize:
        visualize_report(
            mesh_a_local=mesh_a_asset.mesh,
            mesh_b_local=mesh_b_asset.mesh,
            transform_a=optimized_transform_a,
            transform_b=optimized_transform_b,
            report=final_report,
            title="Symmetric SDF Visualization",
        )

    if args.optimize:
        print("\n[Optimized Transforms]")
        _print_matrix("Transform A (optimized)", optimized_transform_a)
        _print_matrix("Transform B (optimized)", optimized_transform_b)

    if args.optimize and optimization_report is not None and not optimization_report.success:
        logger.warning(
            "Joint optimization ended without convergence. stop_reason={}",
            optimization_report.stop_reason,
        )


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
