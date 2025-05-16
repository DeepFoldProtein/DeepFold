"""Hook definitions and default implementations.

Hooks are thin callbacks that expose the state of the pipeline at
well‑defined stages.  Users can override the `BaseHooks` methods
and pass an instance to `~predict.predictor.Predictor`
for custom behaviour (e.g. live logging, visualisation, etc.).
"""

from __future__ import annotations

import gc
import json
import logging
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from deepfold.common import protein
from deepfold.common import residue_constants as rc
from deepfold.utils.file_utils import dump_pickle

logger = logging.getLogger(__name__)


class BaseHooks(Protocol):
    """Typed protocol enumerating the available callbacks."""

    # ------------------------------------------------------------------
    # * Features & batch creation *
    # ------------------------------------------------------------------
    def after_features(self, feats: dict, /, **_: object) -> None: ...

    def after_batch(self, batch: dict, /, **_: object) -> None: ...

    # ------------------------------------------------------------------
    # * Model interaction *
    # ------------------------------------------------------------------
    def after_model_init(self, model: torch.nn.Module, /, **_: object) -> None: ...

    def on_recycle(
        self,
        recycle_iter: int,
        feats: dict,
        outputs: dict,
        /,
        **_: object,
    ) -> None: ...

    def after_prediction(self, outputs: dict, /, **_: object) -> None: ...


class DefaultHooks:  # pragma: no cover – reference implementation mirrors original behaviour
    """Default hooks replicating side‑effects of the legacy script."""

    def __init__(
        self,
        *,
        output_dirpath: Path,
        save_recycle: bool = False,
        suffix: str = "",
        model_name: str = "",
        preset: str = "",
        seed: int = -1,
        benchmark: bool = False,
    ) -> None:
        self.output_dirpath = output_dirpath
        self.save_recycle = save_recycle
        self.suffix = suffix
        self.model_name = model_name
        self.preset = preset
        self.seed = seed
        self.benchmark = benchmark

        if save_recycle:
            self.recycle_out = output_dirpath / f"recycle_{model_name}{suffix}"
            self.recycle_out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Pipeline stages (no‑op unless stated otherwise)
    # ------------------------------------------------------------------
    def after_features(self, feats: dict, **_: object) -> None:  # noqa: D401
        logger.info("input_feature_names=%s", repr(list(feats.keys())))

    def after_batch(self, batch: dict, **_: object) -> None:  # noqa: D401
        """No‑op – extension point."""

    def after_model_init(self, model: torch.nn.Module, **_: object) -> None:  # noqa: D401
        """No‑op – extension point."""

    # ------------------------------------------------------------------
    # Recycling – we largely reproduce the verbose logging & optional PDB
    # ------------------------------------------------------------------
    def on_recycle(self, recycle_iter: int, feats: dict, outputs: dict, **_: object) -> None:  # noqa: D401
        outputs["mean_plddt"] = torch.sum(outputs["plddt"] * feats["seq_mask"]) / torch.sum(feats["seq_mask"])

        pieces: list[str] = []
        for key, label in [
            ("mean_plddt", "plDDT"),
            ("ptm_score", "pTM"),
            ("iptm_score", "ipTM"),
            ("weighted_ptm_score", "Confidence"),
        ]:
            if key in outputs:
                pieces.append(f"{label}={outputs[key]:05.3f}")
        logger.info("Pred: recycle=%d %s", recycle_iter, " ".join(pieces))

        if self.save_recycle:
            batch_np = {
                "residue_index": feats["residue_index"].cpu().squeeze(0).numpy(),
                "aatype": feats["aatype"].cpu().squeeze(0).numpy(),
            }
            if "asym_id" in feats:
                batch_np["asym_id"] = feats["asym_id"].cpu().squeeze(0).numpy()

            outputs_np = {
                "final_atom_mask": outputs["final_atom_mask"].cpu().squeeze(0).numpy(),
                "final_atom_positions": outputs["final_atom_positions"].cpu().squeeze(0).numpy(),
            }

            pdb_path = self.recycle_out / f"frame_{recycle_iter}.pdb"
            prot = protein.from_prediction(
                processed_features=batch_np,
                result=outputs_np,
                b_factors=outputs["plddt"].cpu().squeeze(0).numpy(),
                remark=f"RECYCLE {recycle_iter} {' '.join(pieces)}",
            )
            with open(pdb_path, "w", encoding="utf‑8") as fp:
                fp.write(protein.to_pdb(prot))

            gc.collect()

    # ------------------------------------------------------------------
    # Final summary & result pickle
    # ------------------------------------------------------------------
    def after_prediction(self, outputs: dict, **kwargs: object) -> None:  # noqa: D401
        processed_features: dict = kwargs.pop("processed_features")  # required
        seqlen: int = kwargs.pop("seqlen")  # required
        model_config = kwargs.pop("model_config")
        feat_config = kwargs.pop("feat_config")
        MONOMER_OUTPUT_SHAPES = kwargs.pop("MONOMER_OUTPUT_SHAPES")
        MULTIMER_OUTPUT_SHAPES = kwargs.pop("MULTIMER_OUTPUT_SHAPES")
        unpad_to_schema_shape_ = kwargs.pop("unpad_to_schema_shape_")

        logger.info("Writing unrelaxed PDB + summary ...")
        prot = protein.from_prediction(
            processed_features=processed_features,
            result=outputs,
            b_factors=outputs["plddt"],
            remark=f"{self.preset} seed={self.seed}",
        )
        pdb_path = self.output_dirpath / f"unrelaxed_{self.model_name}{self.suffix}.pdb"
        with open(pdb_path, "w", encoding="utf‑8") as fp:
            fp.write(protein.to_pdb(prot))

        # ------------------------------------------------------------------
        # JSON summary
        # ------------------------------------------------------------------
        summary_path = self.output_dirpath / f"summary_{self.model_name}{self.suffix}.json"
        write_summary(processed_features, outputs, summary_path, self)

        if not self.benchmark:
            logger.info("Serialising full result pickle ...")
            if model_config.is_multimer:
                outputs = unpad_to_schema_shape_(
                    outputs,
                    MULTIMER_OUTPUT_SHAPES,
                    seqlen,
                    feat_config.max_msa_clusters,
                )
            else:
                outputs = unpad_to_schema_shape_(
                    outputs,
                    MONOMER_OUTPUT_SHAPES,
                    seqlen,
                    feat_config.max_msa_clusters,
                )
            dump_pickle(outputs, self.output_dirpath / f"result_{self.model_name}{self.suffix}.pkz")


# ----------------------------------------------------------------------
# Utility – extracted from original _save_summary for re‑use
# ----------------------------------------------------------------------


def write_summary(processed: dict, result: dict, path: Path, hooks: DefaultHooks) -> None:
    """Emit a JSON summary identical to the legacy implementation."""
    seq_mask = processed.get("seq_mask")
    if seq_mask is None:
        seq_mask = np.ones_like(processed["aatype"], dtype=bool)
    else:
        seq_mask = seq_mask.astype(bool)

    summary: dict[str, object] = {
        "model_name": hooks.model_name,
        "preset": hooks.preset,
        "suffix": hooks.suffix,
        "seed": hooks.seed,
        "sequence": "".join(rc.restypes_with_x[i] for i in processed["aatype"][seq_mask]),
        "chain_index": (
            (processed.get("asym_id", np.zeros_like(seq_mask, dtype=np.int64))[seq_mask] - 1).tolist()
            if "asym_id" in processed
            else [0] * int(seq_mask.sum())
        ),
        "residue_index": (processed["residue_index"][seq_mask] + 1).tolist(),
        "plddt": result["plddt"][seq_mask].tolist(),
    }
    # Optional scores
    for key in ("ptm_score", "iptm_score", "weighted_ptm_score"):
        if key in result:
            summary[key.removesuffix("_score")] = float(result[key])

    path.write_text(json.dumps(summary, indent=4), encoding="utf‑8")
