from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AmpConfig(BaseModel):
    """Configuration for automatic mixed precision (torch.autocast)."""

    enabled: bool = Field(
        default=False, title="AMP enabled", description="Enable torch.autocast for model forwards."
    )
    dtype: Literal["float16", "bfloat16"] = Field(
        default="float16",
        title="AMP dtype",
        description="Autocast dtype. Use float16 on Turing GPUs (no bf16).",
    )


class CompileConfig(BaseModel):
    """Configuration for ``torch.compile`` wrapping of the model."""

    enabled: bool = Field(
        default=False,
        title="Compile enabled",
        description="Wrap the encoder (and one-pass decoder, if applicable) with torch.compile.",
    )
    mode: Literal["default", "reduce-overhead", "max-autotune"] = Field(
        default="default",
        title="Compile mode",
        description=(
            "torch.compile mode. 'default' fuses kernels (safe, modest gain). "
            "'reduce-overhead' adds CUDA Graphs (bigger gain, requires fully "
            "static shapes -- requires strict_number_of_instances=False)."
        ),
    )
    fullgraph: bool = Field(
        default=False,
        description="If True, fail loudly on graph breaks instead of falling back to eager.",
    )
