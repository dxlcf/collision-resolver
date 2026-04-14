from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PreprocessingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_unit: Literal["m", "cm", "mm"] = "m"
    default_axis_transform: tuple[tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    coacd_max_convex_parts: int = Field(default=16, ge=1)
    sdf_resolution: int = Field(default=32, ge=8)
    sdf_padding_ratio: float = Field(default=0.1, ge=0.0)
    sdf_sign_method: Literal["ray_casting"] = "ray_casting"
    surface_sample_count: int = Field(default=256, ge=16)
    random_seed: int = 7
    generate_sdf_by_default: bool = True
    generate_surface_samples_by_default: bool = True
    min_triangle_area: float = Field(default=1e-10, gt=0.0)
    min_abs_volume: float = Field(default=1e-12, gt=0.0)


class DetectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    broadphase_padding: float = Field(default=1e-6, ge=0.0)
    contact_limit: int = Field(default=64, ge=1)


class ResolutionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    learning_rate: float = Field(default=0.05, gt=0.0)
    max_iterations: int = Field(default=120, ge=1)
    translation_weight: float = Field(default=0.2, ge=0.0)
    rotation_weight: float = Field(default=0.05, ge=0.0)
    collision_weight: float = Field(default=1.0, gt=0.0)
    safety_margin: float = Field(default=0.01, ge=0.0)
    convergence_tolerance: float = Field(default=1e-4, gt=0.0)
    improvement_tolerance: float = Field(default=1e-5, gt=0.0)
    patience: int = Field(default=8, ge=1)
    max_translation_delta: float = Field(default=0.5, gt=0.0)
    max_rotation_radians: float = Field(default=0.75, gt=0.0)


class VerificationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    require_collision_free: bool = True


class ServiceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    resolution: ResolutionConfig = Field(default_factory=ResolutionConfig)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    reuse_cache: bool = True
