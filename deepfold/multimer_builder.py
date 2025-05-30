"""multimer_builder.py

CLI utility for assembling DeepFold multimer feature pickles from
individual monomer feature files.
"""

from __future__ import annotations

import argparse
import json
import logging
import string
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from deepfold.common import protein
from deepfold.common import residue_constants as rc
from deepfold.data.multimer.input_features import ComplexInfo, process_multimer_features
from deepfold.eval.plot import plot_msa
from deepfold.utils.file_utils import dump_pickle, load_pickle
from deepfold.utils.log_utils import setup_logging

__all__ = [
    "Entity",
    "Structure",
    "parse_recipe",
    "build_features",
    "cli",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Entity:
    """Single asymmetric unit in a multimer."""

    feature_filepath: Path
    num_sym: int = 1
    templates: Optional[dict[str, np.ndarray]] = None


@dataclass(frozen=True, slots=True)
class Structure:
    """Target multimer definition consisting of one or more entities."""

    name: str
    entities: List[Entity] = field(default_factory=list)
    msa_strings: Optional[List[str]] = None
    version: int = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_struct(struct: dict) -> None:
    """Basic validation for one *raw* structure dict.

    Raises
    ------
    ValueError
        If required keys are missing or types are wrong.
    """

    required_root_keys: Sequence[str] = ("name", "entities")
    missing = [k for k in required_root_keys if k not in struct]
    if missing:
        raise ValueError(f"Structure missing keys: {', '.join(missing)}")

    if not isinstance(struct["name"], str):
        raise ValueError("'name' must be a string")

    if not isinstance(struct["entities"], list) or not struct["entities"]:
        raise ValueError("'entities' must be a non‑empty list")

    for ent in struct["entities"]:
        if "path" not in ent:
            raise ValueError("Each entity requires a 'path' key")
        if not isinstance(ent["path"], str):
            raise ValueError("Entity 'path' must be a string")
        if "num_sym" in ent and not isinstance(ent["num_sym"], int):
            raise ValueError("Entity 'num_sym' must be an int if provided")


def _parse_templates(
    templates: list[dict],
    plddt_cutoff: float = 70.0,
) -> dict | None:
    template_aatype = []
    template_all_atom_positions = []
    template_all_atom_mask = []

    for dic in templates:
        pdb_path = Path(dic["pdb_path"])
        pdb_str = pdb_path.read_text()
        prot = protein.from_pdb_string(pdb_str, dic["chain_id"])

        seq = rc.aatype_to_str_sequence(prot.aatype)
        aatype = rc.sequence_to_onehot(seq, rc.HHBLITS_AA_TO_ID)

        mask = prot.atom_mask * (prot.b_factors > plddt_cutoff)
        if "regions" in dic:
            for low, high in dic["regions"]:
                mask[low - 1 : high, :] *= 1

        if mask[:, 1].sum() == 0:
            logger.info("No template parsed from %s", pdb_path)
            continue

        template_all_atom_positions.append(prot.atom_positions)
        template_all_atom_mask.append(mask)
        template_aatype.append(aatype)

    if len(template_all_atom_mask):
        return {
            "template_aatype": np.stack(template_aatype),
            "template_all_atom_positions": np.stack(template_all_atom_positions),
            "template_all_atom_mask": np.stack(template_all_atom_mask),
        }
    else:
        logger.warning("BE CAREFUL! No template parsed")
        return None


def parse_recipe(recipe: str | Path) -> List[Structure]:
    """Parse and validate a JSON‑encoded multimer recipe.

    Parameters
    ----------
    recipe
        Path to the JSON file **or** a JSON string.

    Returns
    -------
    list[Structure]
    """

    raw_json: str
    if isinstance(recipe, (str, Path)) and Path(recipe).is_file():
        raw_json = Path(recipe).read_text()
    elif isinstance(recipe, str):
        raw_json = recipe
    else:
        raise TypeError("'recipe' must be a path‑like object or a JSON string")

    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON: {exc}") from exc

    if not isinstance(data, dict) or "structures" not in data:
        raise ValueError("JSON must contain a top‑level 'structures' list")

    structures: List[Structure] = []
    for struct in data["structures"]:
        _validate_struct(struct)

        if "msa_path" in struct:
            msa_strings = Path(struct["msa_path"]).read_text().strip(" \n\r\t\x00").split("\x00")
        else:
            msa_strings = None

        entities = []
        for entity in struct["entities"]:
            if "templates" in entity:
                template_feats = _parse_templates(entity["templates"])
            else:
                template_feats = None
            entities.append(
                Entity(
                    feature_filepath=Path(entity["path"]),
                    num_sym=int(entity.get("num_sym", 1)),
                    templates=template_feats,
                )
            )

        structures.append(Structure(name=struct["name"], entities=entities, msa_strings=msa_strings))

    return structures


def build_features(
    structure: Structure,
    *,
    output_root: Path,
    source_root: Path,
    skip_plot: bool = False,
    parse_pair_descr: bool = False,
) -> Dict[str, str]:
    """Generate DeepFold multimer features and (optionally) an MSA‑depth plot."""

    if structure.version != 2:
        raise ValueError(f"Unsupported structure.version {structure.version!r}")

    out_dir = output_root / structure.name
    out_dir.mkdir(parents=True, exist_ok=True)

    descriptions: List[str] = []
    num_units: List[int] = []
    monomer_features: Dict[str, dict] = {}
    stoichiometry_parts: List[str] = []
    paired_msas: Dict[str, str] = {}

    # ---------- Load and validate monomer feature files ----------
    override_msas = structure.msa_strings is not None

    # for chain_id, entity, msa in zip(string.ascii_uppercase, structure.entities, msa_strings):
    for chain_id, index in zip(string.ascii_uppercase, range(len(structure.entities))):
        entity = structure.entities[index]
        msa = structure.msa_strings[index] if override_msas else ""

        monomer_path = source_root / entity.feature_filepath
        if not monomer_path.exists():
            raise FileNotFoundError(f"Monomer feature file not found: {monomer_path}")

        descriptions.append(chain_id)
        num_units.append(entity.num_sym)
        stoichiometry_parts.append(f"{chain_id}{entity.num_sym}")

        feats: dict[str, np.ndarray] = load_pickle(monomer_path)

        if entity.templates is not None:
            for key in [
                "template_domain_names",
                "template_sequence",
                "template_aatype",
                "template_all_atom_positions",
                "template_all_atom_mask",
                "template_sum_probs",
            ]:
                if key in feats:
                    del feats[key]

            feats.update(entity.templates)

        monomer_features[chain_id] = feats

        msa = msa.strip()
        if msa:
            paired_msas[chain_id] = msa

    complex_info = ComplexInfo(descriptions=descriptions, num_units=num_units)
    combined_features = process_multimer_features(
        complex=complex_info,
        all_monomer_features=monomer_features,
        paired_a3m_strings=paired_msas,
        use_paired_a3m_descr=parse_pair_descr,
    )

    # Uniform sampling
    if not parse_pair_descr:
        combined_features.pop("msa_weight")

    for k, v in combined_features.items():
        logger.info(f"{k}: {v.dtype}{list(v.shape)}")

    dump_pickle(combined_features, out_dir / "features.pkz")

    if not skip_plot:
        fig = plot_msa(combined_features)
        fig.savefig(out_dir / "msa_depth.png")
        plt.close(fig)

    return {
        "name": structure.name,
        "stoichiometry": "".join(stoichiometry_parts),
        "output_dir": str(out_dir.resolve()),
    }


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def _example_recipe() -> str:
    example = {
        "structures": [
            {
                "name": "multimer",
                "entities": [
                    {
                        "path": "features/chainA.pkz",
                        "num_sym": 2,
                        "templates": [
                            {
                                "pdb_path": "pred/chainA/model_3.pdb",
                                "chain_id": "A",
                                "regions": [[1, 45], [60, 98]],
                            },
                        ],
                    },
                    {"path": "features/chainB.pkz", "num_sym": 3},
                ],
                "msa_path": "msas/paired.a3m",
            }
        ]
    }
    return json.dumps(example, indent=2)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build DeepFold multimer features from per‑monomer feature files.",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        help="Path to recipe JSON.",
    )
    parser.add_argument(
        "-s",
        "--source-dir",
        type=Path,
        help="Directory containing monomer feature files (defaults to recipe parent)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory in which to write outputs (defaults to CWD)",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Do not generate msa_depth.png",
    )
    parser.add_argument(
        "--show-example",
        action="store_true",
        help="Print example recipe and exit",
    )
    parser.add_argument(
        "--parse-pair-descr",
        dest="parse_pair_descr",
        action="store_true",
        help="Parse alignment score from the descriptions",
    )
    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def cli(argv: List[str] | None = None) -> None:  # pragma: no cover
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.show_example:
        print(_example_recipe())
        sys.exit(0)

    if args.input is None:
        parser.error("--input is required unless --show-example is given.")

    setup_logging("multimer.log")

    logger.info("Reading recipe from %s", args.input)
    try:
        structures = parse_recipe(args.input)
    except Exception as exc:  # noqa: BLE001 (broad except is fine for CLI)
        logger.error("Failed to parse recipe: %s", exc)
        sys.exit(1)

    source_dir = args.source_dir or args.input.parent
    logger.info("Using source dir: %s", source_dir)

    # Show debug params
    for key, value in vars(args).items():
        logger.debug("%s = %s", key, value)

    for struct in tqdm(structures, desc="Building multimers", unit="structure"):
        try:
            logger.info("Build multimer: %s", struct.name)
            meta = build_features(
                struct,
                output_root=args.output_dir,
                source_root=source_dir,
                skip_plot=args.skip_plot,
                parse_pair_descr=args.parse_pair_descr,
            )
            logger.info("Built multimer: %s", meta)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to build %s: %s", struct.name, exc, exc_info=True)


if __name__ == "__main__":  # pragma: no cover
    cli()
