"""Preset mapping and helper utilities."""

from __future__ import annotations

from typing import Any, Tuple

__all__ = [
    "get_preset_opts",
    "VALID_PRESETS",
]

# ---------------------------------------------------------------------
# Preset lookup table – kept identical to the original script for parity.
# ---------------------------------------------------------------------
_PRESET_MAP: dict[str, str] = {
    # DeepFold (training code‑name) models
    **{f"deepfold_model_{i}": f"model_{i}" for i in range(1, 6)},
    # Parameter archives (AlphaFold‑style)
    **{f"params_model_{i}": f"model_{i}" for i in range(1, 6)},
    **{f"params_model_{i}_ptm": f"model_{i}_ptm" for i in range(1, 6)},
    # Multimers – v1/v2/v3
    **{f"params_model_{i}_multimer": f"model_{i}_v1" for i in range(1, 6)},
    **{f"params_model_{i}_multimer_v2": f"model_{i}_v2" for i in range(1, 6)},
    **{f"params_model_{i}_multimer_v3": f"model_{i}_v3" for i in range(1, 6)},
}

VALID_PRESETS: tuple[str, ...] = tuple(_PRESET_MAP.keys())


def _bool_from_preset(name: str, key: str) -> bool:
    """Return *True* if *key* is present in *name* (utility)."""
    return key in name


def get_preset_opts(preset: str) -> Tuple[str, tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
    """Map *preset* string onto model‑name and configuration kwargs.

    The second return value is a 3‑tuple of kwargs dictionaries for
    ``AlphaFoldConfig``, ``FeatureConfig`` and ``import_jax_weights_``.
    """
    model_name = _PRESET_MAP.get(preset, preset)  # Fallback to raw string.

    is_multimer = _bool_from_preset(preset, "multimer")
    enable_ptm = _bool_from_preset(preset, "ptm") or is_multimer
    enable_templates = not any(preset.endswith(x) for x in ["_3", "_4", "_5"]) or is_multimer
    fuse_projection_weights = preset.endswith("multimer_v3")

    model_cfg_kwargs: dict[str, Any] = dict(
        is_multimer=is_multimer,
        enable_ptm=enable_ptm,
        enable_templates=enable_templates,
        inference_chunk_size=4,
        inference_block_size=256,
    )
    feat_cfg_kwargs: dict[str, Any] = dict(is_multimer=is_multimer)
    import_kwargs: dict[str, Any] = dict(
        is_multimer=is_multimer,
        enable_ptm=enable_ptm,
        enable_templates=enable_templates,
        fuse_projection_weights=fuse_projection_weights,
    )

    if preset.startswith("deepfold"):
        feat_cfg_kwargs["max_recycling_iters"] = 10

    return model_name, (model_cfg_kwargs, feat_cfg_kwargs, import_kwargs)
