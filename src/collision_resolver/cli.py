from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import open3d as o3d
from loguru import logger

from collision_resolver.sdf_collision import (
    CollisionReport,
    DirectionCollisionResult,
    ResolveReport,
    detect_collision,
    load_mesh,
    resolve_collision_by_translation,
    resolve_runtime_parameters,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use SDF to detect collision between two triangle meshes.",
    )
    parser.add_argument("mesh_a", help="Path of the first mesh file.")
    parser.add_argument("mesh_b", help="Path of the second mesh file.")
    parser.add_argument(
        "--offset",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.0),
        metavar=("DX", "DY", "DZ"),
        help="Optional translation applied to the second mesh.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help=(
            "Explicit override for the number of surface samples used for each direction. "
            "If omitted, the count is derived from --sample-spacing-ratio."
        ),
    )
    parser.add_argument(
        "--sample-spacing-ratio",
        type=float,
        default=5e-3,
        help=(
            "Target surface sampling spacing as a ratio of the combined scene bbox diagonal. "
            "Used to derive per-mesh sample counts when --samples is omitted."
        ),
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=5000,
        help="Lower bound for the auto-derived per-mesh sample count.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200000,
        help="Upper bound for the auto-derived per-mesh sample count.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize the meshes, SDF penetration points, and collision bbox.",
    )
    parser.add_argument(
        "--resolve",
        action="store_true",
        help="Automatically translate mesh B along the SDF-gradient push direction.",
    )
    parser.add_argument(
        "--resolve-visualize",
        action="store_true",
        help="When resolving, visualize both the pre-resolve and post-resolve states.",
    )
    parser.add_argument(
        "--gradient-eps",
        type=float,
        default=None,
        help=(
            "Explicit finite-difference epsilon used to estimate the SDF gradient. "
            "If omitted, derived from --gradient-eps-ratio."
        ),
    )
    parser.add_argument(
        "--gradient-eps-ratio",
        type=float,
        default=1e-4,
        help="Gradient epsilon as a ratio of the combined scene bbox diagonal.",
    )
    parser.add_argument(
        "--max-resolve-iters",
        type=int,
        default=20,
        help="Maximum number of translation iterations for collision resolution.",
    )
    parser.add_argument(
        "--safety-margin",
        type=float,
        default=None,
        help=(
            "Explicit extra translation distance added on top of the current penetration depth. "
            "If omitted, derived from --safety-margin-ratio."
        ),
    )
    parser.add_argument(
        "--safety-margin-ratio",
        type=float,
        default=1e-5,
        help="Safety margin as a ratio of the combined scene bbox diagonal.",
    )
    parser.add_argument(
        "--output-resolved-mesh",
        type=str,
        default=None,
        help="Write the resolved mesh B to this path after --resolve finishes.",
    )
    return parser.parse_args()


def format_vec(vec: np.ndarray | None) -> str:
    if vec is None:
        return "N/A"
    return f"[{vec[0]:.6f}, {vec[1]:.6f}, {vec[2]:.6f}]"


def print_direction_result(result: DirectionCollisionResult) -> None:
    print(f"\n[{result.label}]")
    print(f"Collision: {'YES' if result.collision else 'NO'}")
    print(f"Penetrating samples: {result.penetrating_count} / {result.sample_count}")
    print(f"Max penetration depth: {result.max_penetration_depth:.6f}")
    print(f"BBox min: {format_vec(result.bbox_min)}")
    print(f"BBox max: {format_vec(result.bbox_max)}")


def o3d_mesh_to_pv(mesh: o3d.geometry.TriangleMesh, pv_module):
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    faces = np.hstack(
        [np.full((len(triangles), 1), 3, dtype=np.int64), triangles.astype(np.int64)],
    )
    return pv_module.PolyData(vertices, faces)


def visualize_collision(
    mesh_a: o3d.geometry.TriangleMesh,
    mesh_b: o3d.geometry.TriangleMesh,
    bbox_min: np.ndarray | None,
    bbox_max: np.ndarray | None,
    results: list[DirectionCollisionResult],
    title: str = "SDF Collision Visualization",
) -> None:
    try:
        import pyvista as pv
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Visualization requires pyvista. Install optional dependency first.",
        ) from exc

    plotter = pv.Plotter(title=title)

    mesh_a_pv = o3d_mesh_to_pv(mesh_a, pv)
    mesh_b_pv = o3d_mesh_to_pv(mesh_b, pv)
    plotter.add_mesh(mesh_a_pv, color="cyan", opacity=0.2, label="Mesh A")
    plotter.add_mesh(mesh_b_pv, color="magenta", opacity=0.2, label="Mesh B")

    all_points = [r.penetration_points for r in results if len(r.penetration_points) > 0]
    if all_points:
        merged_points = np.vstack(all_points)
        plotter.add_points(
            merged_points,
            color="yellow",
            point_size=4,
            render_points_as_spheres=True,
            label="SDF Penetration Points",
        )

    if bbox_min is not None and bbox_max is not None:
        bbox_mesh = pv.Box(
            [
                bbox_min[0],
                bbox_max[0],
                bbox_min[1],
                bbox_max[1],
                bbox_min[2],
                bbox_max[2],
            ],
        )
        plotter.add_mesh(
            bbox_mesh.scale(1.01),
            color="lime",
            style="wireframe",
            line_width=6,
            label="BBox",
        )

    plotter.add_legend()
    plotter.show()


def print_header(
    args: argparse.Namespace,
    offset: np.ndarray,
    collision_before: CollisionReport,
    runtime,
) -> None:
    print("=" * 72)
    print(f"{'SDF COLLISION DETECTION':^72}")
    print("=" * 72)
    print(f"Mesh A: {Path(args.mesh_a)}")
    print(f"Mesh B: {Path(args.mesh_b)}")
    print(f"Mesh B offset: [{offset[0]:.6f}, {offset[1]:.6f}, {offset[2]:.6f}]")
    print(f"Scene bbox diagonal: {runtime.scene_scale:.6f}")
    print(f"Target sample spacing: {runtime.target_spacing:.6f}")
    print(
        "Samples per direction: "
        f"A={runtime.sample_count_a}, B={runtime.sample_count_b}",
    )
    print(f"Gradient epsilon: {runtime.gradient_eps:.6f}")
    print(f"Safety margin: {runtime.safety_margin:.6f}")
    print(f"Overall collision: {'YES' if collision_before.overall_collision else 'NO'}")
    print(f"Overall max penetration depth: {collision_before.overall_depth:.6f}")
    print(f"Overall bbox min: {format_vec(collision_before.overall_bbox_min)}")
    print(f"Overall bbox max: {format_vec(collision_before.overall_bbox_max)}")


def print_resolve_result(
    resolve_result: ResolveReport,
    collision_after: CollisionReport,
) -> None:
    resolved_by_post_check = resolve_result.resolved and not collision_after.overall_collision

    print("\n[Collision Resolution]")
    print(f"Resolved: {'YES' if resolved_by_post_check else 'NO'}")
    print(f"Iterations: {resolve_result.iterations}")
    print(
        "Initial push direction: "
        f"{format_vec(resolve_result.first_direction) if resolve_result.first_direction is not None else 'N/A'}",
    )
    print(
        "Initial suggested translation: "
        f"{format_vec(resolve_result.first_step) if resolve_result.first_step is not None else 'N/A'}",
    )
    print(f"Total translation applied: {format_vec(resolve_result.total_translation)}")
    print(f"Post-resolve collision: {'YES' if collision_after.overall_collision else 'NO'}")
    print(f"Post-resolve max penetration depth: {collision_after.overall_depth:.6f}")

    if resolve_result.resolved and not resolved_by_post_check:
        logger.warning(
            "Resolver converged but post-check still reports collision. "
            "Consider increasing --samples or --max-resolve-iters.",
        )


def run(args: argparse.Namespace) -> None:
    if args.resolve_visualize:
        args.resolve = True
    if args.output_resolved_mesh is not None:
        args.resolve = True

    mesh_a = load_mesh(args.mesh_a)
    mesh_b = load_mesh(args.mesh_b)

    offset = np.asarray(args.offset, dtype=float)
    if np.any(offset != 0.0):
        logger.info("Applying translation to mesh B: {}", offset.tolist())
        mesh_b.translate(offset)

    runtime = resolve_runtime_parameters(
        mesh_a,
        mesh_b,
        samples=args.samples,
        sample_spacing_ratio=args.sample_spacing_ratio,
        min_samples=args.min_samples,
        max_samples=args.max_samples,
        gradient_eps=args.gradient_eps,
        gradient_eps_ratio=args.gradient_eps_ratio,
        safety_margin=args.safety_margin,
        safety_margin_ratio=args.safety_margin_ratio,
    )

    mesh_b_before_resolve = copy.deepcopy(mesh_b) if args.resolve_visualize else None
    collision_before = detect_collision(
        mesh_a,
        mesh_b,
        runtime.sample_count_a,
        runtime.sample_count_b,
    )

    print_header(args, offset, collision_before, runtime)
    print_direction_result(collision_before.result_b_in_a)
    print_direction_result(collision_before.result_a_in_b)

    collision_after = collision_before
    resolve_result = None

    if args.resolve:
        resolve_result = resolve_collision_by_translation(
            mesh_a=mesh_a,
            mesh_b=mesh_b,
            sample_count_a=runtime.sample_count_a,
            sample_count_b=runtime.sample_count_b,
            eps=runtime.gradient_eps,
            max_iters=args.max_resolve_iters,
            safety_margin=runtime.safety_margin,
        )
        collision_after = detect_collision(
            mesh_a,
            mesh_b,
            runtime.sample_count_a,
            runtime.sample_count_b,
        )
        print_resolve_result(resolve_result, collision_after)

        if args.output_resolved_mesh is not None:
            output_path = Path(args.output_resolved_mesh)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            success = o3d.io.write_triangle_mesh(str(output_path), mesh_b)
            if not success:
                raise RuntimeError(f"Failed to write resolved mesh: {output_path}")
            print(f"Resolved mesh written to: {output_path}")

    print("=" * 72)

    if args.resolve_visualize and mesh_b_before_resolve is not None:
        visualize_collision(
            mesh_a=mesh_a,
            mesh_b=mesh_b_before_resolve,
            bbox_min=collision_before.overall_bbox_min,
            bbox_max=collision_before.overall_bbox_max,
            results=[collision_before.result_b_in_a, collision_before.result_a_in_b],
            title="SDF Collision Visualization - Before Resolve",
        )
        visualize_collision(
            mesh_a=mesh_a,
            mesh_b=mesh_b,
            bbox_min=collision_after.overall_bbox_min,
            bbox_max=collision_after.overall_bbox_max,
            results=[collision_after.result_b_in_a, collision_after.result_a_in_b],
            title="SDF Collision Visualization - After Resolve",
        )
    elif args.visualize:
        visualize_collision(
            mesh_a=mesh_a,
            mesh_b=mesh_b,
            bbox_min=collision_after.overall_bbox_min,
            bbox_max=collision_after.overall_bbox_max,
            results=[collision_after.result_b_in_a, collision_after.result_a_in_b],
        )


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
