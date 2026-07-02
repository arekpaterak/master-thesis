from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from core.models.nds.encoders.config import EncoderConfig


class OriginalPretrainedConfig(BaseModel):
    """Configuration for initialising from an original NDS pretrained checkpoint."""

    path: str = Field(
        title="Original pretrained checkpoint",
        description="Path to the pretrained checkpoint file.",
    )
    load_decoder: bool = Field(
        default=False,
        title="Load decoder from original pretrained",
        description="Also load decoder weights.",
    )


# Encoder settings that reproduce the released NDS implementation, which produced the
# original pretrained checkpoints.
ORIGINAL_NDS_ENCODING_ORDER: Literal["route_neighbours"] = "route_neighbours"
ORIGINAL_NDS_ROUTE_AGGREGATION: Literal["sum"] = "sum"


def validate_encoder_matches_original_nds(encoder: EncoderConfig) -> None:
    """Raise unless the encoder graph matches the original NDS checkpoints.

    Neither field is a weight, so a mismatched graph loads silently and computes a
    different function. Configs that load an original checkpoint must set both explicitly
    rather than relying on the (non-pretrained) defaults.
    """
    for field, required in (
        ("encoding_order", ORIGINAL_NDS_ENCODING_ORDER),
        ("route_aggregation", ORIGINAL_NDS_ROUTE_AGGREGATION),
    ):
        current = getattr(encoder, field)
        if current != required:
            raise ValueError(
                f"original NDS pretrained weights require encoder.{field}={required!r}, "
                f"got {current!r}; set it explicitly in the config."
            )
