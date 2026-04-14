from __future__ import annotations

import argparse

from collision_resolver.config import ServiceConfig
from collision_resolver.exceptions import BackendUnavailableError
from collision_resolver.service import CollisionResolverService


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="collision-resolver",
        description=(
            "当前 CLI 仅执行运行时依赖自检并打印默认配置，不接收网格输入，也不输出碰撞检测或修正结果。"
            "业务调用请通过 Python API 构造 ServiceRequest，并读取 ServiceResult。"
        ),
    )
    parser.parse_args(argv)

    config = ServiceConfig()
    try:
        CollisionResolverService(config)
    except BackendUnavailableError as exc:
        raise SystemExit(f"{exc.message} [code={exc.code}]") from exc

    print(
        "collision-resolver 运行时检查通过。"
        " 当前 CLI 仅执行依赖自检，不接收网格输入，也不会输出碰撞结果。"
        " 业务调用请使用 Python API 中的 ServiceRequest -> ServiceResult 流程。"
        f" 当前默认配置: sdf_resolution={config.preprocessing.sdf_resolution},"
        f" sample_count={config.preprocessing.surface_sample_count},"
        f" optimizer_iterations={config.resolution.max_iterations}。"
    )
