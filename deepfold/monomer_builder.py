"""Builder for DeepFold input feature pickles."""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from deepfold.common import protein
from deepfold.common import residue_constants as rc
from deepfold.data.search.crfalign import parse_crf
from deepfold.data.search.input_features import (
    create_msa_features,
    create_sequence_features,
    create_template_features,
)
from deepfold.data.search.parsers import (
    convert_stockholm_to_a3m,
    parse_fasta,
    parse_hhr,
    parse_hmmsearch_sto,
)
from deepfold.data.search.templates import (
    TemplateHitFeaturizer,
    create_empty_template_feats,
)
from deepfold.utils.file_utils import (
    dump_pickle,
    get_file_content_and_extension,
    load_pickle,
)
from deepfold.utils.log_utils import setup_logging

__all__ = [
    "Domain",
    "parse_dom",
    "build_input_features",
    "cli",
]

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataclasses & helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Domain:
    """Represents a query subsequence to be replaced by a template model."""

    doi: int  # domain order index (1‑based per DeepFold convention)
    target_start: int  # **0‑based inclusive** index into query sequence
    target_end: int  # **0‑based exclusive** index into query sequence
    model_name: str  # path to PDB or .pkz or identifier used to derive path
    result_start: int | None  # slice of template result to use
    result_end: int | None
    chain_id: str | None = None  # which chain to take when reading PDB

    # Convenience ----------------------------------------------------------------

    @property
    def span(self) -> slice:  # noqa: D401 (property is fine)
        """Return *Pythonic* slice for the domain in query space."""
        return slice(self.target_start, self.target_end)


# ---------------------------------------------------------------------------
# Parsing utilities
# ---------------------------------------------------------------------------


_DOM_LINE_RE = re.compile(
    r"^(?P<doi>\d+)\s+"  # domain index or 0 for CRF codes
    r"(?P<span>\d+-\d+)\s+"  # target span (start‑end, 1‑based closed)
    r"(?P<model>\S+)"  # model filename or identifier
    r"(?:\s+(?P<chain>[^\s-]+)|\s+(?P<crop>\d+-\d+))?"  # optional chain id *or* crop
    r"\s*$",
)


def parse_dom(dom_str: str) -> Tuple[List[Domain], List[str]]:
    """Parse DeepFold "dom" format.

    The *dom* format mixes domain specifications and, optionally, CRFalign
    chunks (lines starting with `0`).

    Returns
    -------
    domains
        Ordered list of :class:`Domain` objects.
    crf_codes
        Any CRFalign alignment strings (in the original order) so they can
        be passed downstream.
    """

    cleaned = [seg.partition("#")[0].strip() for seg in dom_str.splitlines()]
    cleaned = [line for line in cleaned if line]

    domains: List[Domain] = []
    crf_codes: List[str] = []

    for line in cleaned:
        parts = line.split()
        doi = int(parts[0])
        if doi == 0:
            crf_codes.extend(parts[1:])
            continue

        match = _DOM_LINE_RE.match(line)
        if not match:
            raise ValueError(f"Malformed dom line: {line!r}")

        span_1based = match["span"]
        start_1, end_1 = map(int, span_1based.split("-"))
        # Convert to 0‑based half‑open
        start, end = start_1 - 1, end_1

        if match["crop"]:
            raise NotImplementedError("Domain cropping is not supported yet.")
        chain = match["chain"] if match["chain"] else None

        domains.append(
            Domain(
                doi=doi,
                target_start=start,
                target_end=end,
                model_name=match["model"],
                result_start=None,
                result_end=None,
                chain_id=chain,
            )
        )

    return domains, crf_codes


# ---------------------------------------------------------------------------
# Template domain feature builder
# ---------------------------------------------------------------------------


def _domain_to_template_features(
    domains: Sequence[Domain],
    query_name: str,
    query_seq: str,
) -> dict:
    """Convert explicit domain‑replacement instructions to template features."""

    if not domains:
        return create_empty_template_feats(len(query_seq), empty=True)

    out: dict[str, list] = {
        "template_domain_names": [],
        "template_sequence": [],
        "template_aatype": [],
        "template_all_atom_positions": [],
        "template_all_atom_mask": [],
        "template_sum_probs": [],
    }

    seqlen = len(query_seq)
    query_aatype = rc.sequence_to_onehot(query_seq, rc.HHBLITS_AA_TO_ID)

    for dom in domains:
        # ------------------------------------------------------------------ PDB
        if Path(dom.model_name).suffix == ".pdb":
            pdb_path = Path(dom.model_name)
            if not pdb_path.exists():
                raise FileNotFoundError(pdb_path)

            pdb_str, _ = get_file_content_and_extension(pdb_path)
            prot = protein.from_pdb_string(pdb_str, chain_id=dom.chain_id)

            index_map = prot.residue_index - prot.residue_index.min()
            residue_index = np.arange(
                prot.residue_index.min(),
                prot.residue_index.max() + 1,
                dtype=np.int32,
            )
            atom_pos = np.zeros((len(residue_index), rc.atom_type_num, 3), dtype=np.float32)
            atom_mask = np.zeros((len(residue_index), rc.atom_type_num), dtype=np.float32)
            atom_pos[index_map] = prot.atom_positions
            atom_mask[index_map] = prot.atom_mask

            dom_seq = rc.aatype_to_str_sequence(prot.aatype)
            if dom_seq != query_seq[dom.target_start : dom.target_end]:
                log.warning("Sequence mismatch for %s: %s", dom.model_name, dom_seq)

            if len(residue_index) != dom.target_end - dom.target_start:
                raise ValueError("Residue length mismatch for domain %s" % dom.model_name)

            pad_left = dom.target_start
            pad_right = seqlen - dom.target_end
            pos_padded = np.pad(atom_pos, ((pad_left, pad_right), (0, 0), (0, 0)))
            mask_padded = np.pad(atom_mask, ((pad_left, pad_right), (0, 0)))

        # --------------------------------------------------------------- Result pkz
        else:
            path = Path(dom.model_name)
            if path.suffix not in {".pkz", ".pkl"}:
                # infer location from query name and doi per DeepFold convention
                sub_parts = dom.model_name.split("/")
                path = Path("/".join([f"{query_name}_{dom.doi}"] + sub_parts[:-1] + [f"result_{sub_parts[-1]}.pkz"]))
            if not path.exists():
                raise FileNotFoundError(path)

            results = load_pickle(path)
            i1 = dom.result_start or 0
            i2 = dom.result_end or (dom.target_end - dom.target_start)
            if (i2 - i1) != (dom.target_end - dom.target_start):
                raise ValueError("Result slice length mismatch for %s" % dom.model_name)

            pos_padded = np.pad(
                results["final_atom_positions"][i1:i2],
                ((dom.target_start, seqlen - dom.target_end), (0, 0), (0, 0)),
            )
            mask_padded = np.pad(
                results["final_atom_mask"][i1:i2],
                ((dom.target_start, seqlen - dom.target_end), (0, 0)),
            )

        # ---------------------------------------------------------- Append to list
        log.info(
            "[doi=%d] %-40s → padded (%d,%d,%d)",
            dom.doi,
            dom.model_name,
            *pos_padded.shape,
        )

        out["template_domain_names"].append(dom.model_name.encode())
        out["template_sequence"].append(query_seq.encode())
        out["template_aatype"].append(query_aatype.copy())
        out["template_all_atom_positions"].append(pos_padded)
        out["template_all_atom_mask"].append(mask_padded)
        out["template_sum_probs"].append(1.0)

    # ------------------------------------------------------------------ Stack
    stacked = {
        "template_domain_names": np.asarray(out["template_domain_names"], dtype=np.object_),
        "template_sequence": np.asarray(out["template_sequence"], dtype=np.object_),
        "template_aatype": np.asarray(out["template_aatype"], dtype=np.int32),
        "template_all_atom_positions": np.stack(out["template_all_atom_positions"], dtype=np.float32),
        "template_all_atom_mask": np.stack(out["template_all_atom_mask"], dtype=np.float32),
        "template_sum_probs": np.asarray(out["template_sum_probs"], dtype=np.float32).reshape(-1, 1),
    }
    return stacked


# ---------------------------------------------------------------------------
# Core public builder
# ---------------------------------------------------------------------------


def build_input_features(
    *,
    fasta_path: Path,
    alignment_paths: Sequence[Path] | None = None,
    template_path: Path | None = None,
    output_path: Path,
    pdb_mmcif_dir: Path | None = None,
    pdb_obsolete_path: Path | None = None,
    crf_alignment_dir: Path | None = None,
    kalign_bin: str = "kalign",
    max_template_hits: int = 20,
    max_template_date: str | None = None,
    template_mode: str = "auto",
    seed: int | None = None,
    offset: int = 0,
) -> None:
    """Standalone function that reproduces the CLI behaviour."""

    # ---------------------------------------------------------------------- FASTA
    fasta_str, _ = get_file_content_and_extension(fasta_path)
    seqs, descs = parse_fasta(fasta_str)
    if len(seqs) != 1:
        raise ValueError("FASTA must contain exactly one sequence")

    query_seq = seqs[0]
    desc = descs[0]
    query_name = desc.split()[0]
    seqlen = len(query_seq)

    log.info("Building features for %s (len=%d)", query_name, seqlen)

    # ---------------------------------------------------------------- Sequence
    seq_feats = create_sequence_features(query_seq, query_name)
    if offset:
        log.info("Applying residue index offset=%d", offset)
        seq_feats["residue_index"] += offset

    # ---------------------------------------------------------------- Templates
    tmpl_feats = create_empty_template_feats(seqlen)
    domains: List[Domain] | None = None

    if template_path is not None:
        log.info("Reading template hits from %s", template_path)
        tmpl_str, suffix = get_file_content_and_extension(template_path)

        if template_mode == "auto":
            if suffix == ".sto":
                template_mode = "hhm"
            elif suffix == ".hhr":
                template_mode = "hhr"
            elif suffix == ".crf":
                template_mode = "crf"
            else:
                raise RuntimeError(f"Unsupported template file extension: {suffix}")

        template_featurizer = TemplateHitFeaturizer(
            max_template_hits=max_template_hits,
            pdb_mmcif_dirpath=pdb_mmcif_dir,
            kalign_executable_path=kalign_bin,
            verbose=True,
        )

        sort_by_sum_probs = template_mode not in {"crf", "dom"}
        template_hits = []

        if template_mode == "hhm":
            template_hits = parse_hmmsearch_sto(query_seq, tmpl_str)
        elif template_mode == "hhr":
            template_hits = parse_hhr(tmpl_str)
        elif template_mode == "crf":
            if crf_alignment_dir is None:
                raise ValueError("--crf-alignment-dirpath is required for CRF mode")
            template_hits = parse_crf(tmpl_str, query_id=query_name, alignment_dir=crf_alignment_dir)
            sort_by_sum_probs = False
        elif template_mode == "dom":
            domains, crf_codes = parse_dom(tmpl_str)
            tmpl_feats = create_empty_template_feats(seqlen, empty=True)
            if crf_codes:
                template_hits = parse_crf("\n".join(crf_codes), query_id=query_name, alignment_dir=crf_alignment_dir)
            sort_by_sum_probs = False
        else:
            raise RuntimeError(f"Unknown template_mode '{template_mode}'")

        if template_hits:
            tmpl_feats = create_template_features(
                sequence=query_seq,
                template_hits=template_hits,
                template_hit_featurizer=template_featurizer,
                max_release_date=max_template_date or datetime.today().strftime("%Y-%m-%d"),
                sort_by_sum_probs=sort_by_sum_probs,
                shuffling_seed=seed,
            )

    # ----------------------------- Additional domain‑based templates ("dom")
    if domains:
        more_tmpl = _domain_to_template_features(domains, query_name, query_seq)
        for k in tmpl_feats.keys():
            tmpl_feats[k] = np.concatenate([tmpl_feats[k], more_tmpl[k]], axis=0)

    # ------------------------------------------------------------------- MSA
    a3m_strings: List[str] = []
    max_num_seqs = {
        "bfd_uniclust_hits": None,
        "mgnify_hits": 5000,
        "uniref90_hits": 10000,
        "uniprot_hits": 50000,
    }

    if alignment_paths:
        log.info("Parsing %d alignment files", len(alignment_paths))
        for path in tqdm(alignment_paths, desc="MSA files", unit="file"):
            a3m_str, suffix = get_file_content_and_extension(path)
            if suffix == ".a3m":
                pass
            elif suffix == ".sto":
                key = path.stem.split(".")[0]
                a3m_str = convert_stockholm_to_a3m(a3m_str, max_sequences=max_num_seqs.get(key))
            else:
                raise RuntimeError(f"Unsupported alignment extension: {suffix}")
            a3m_strings.append(a3m_str)

    msa_feats = create_msa_features(a3m_strings, sequence=query_seq, use_identifiers=True)

    # ------------------------------------------------------------------- Dump
    log.info("Writing feature pickle to %s", output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dump_pickle({**seq_feats, **msa_feats, **tmpl_feats}, output_path, level=5)


# ---------------------------------------------------------------------------
# CLI & entry‑point
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DeepFold search feature builder")
    p = parser.add_argument
    p("-f", "--fasta", type=Path, required=True, help="Input FASTA")
    p("-a", "--alignments", type=Path, nargs="*", help="MSA search result files (*.a3m|*.sto)")
    p("-t", "--template", type=Path, help="Template search result file")
    p("-o", "--output", type=Path, required=True, help="Output .pkz")

    p("--pdb-mmcif-dir", type=Path, help="Directory with gzipped mmCIFs")
    p("--pdb-obsolete", type=Path, help="File listing obsolete PDB entries")
    p("--max-template-date", default=datetime.today().strftime("%Y-%m-%d"))
    p("--max-template-hits", type=int, default=20)
    p("--template-mode", default="auto", choices=["auto", "hhr", "hhm", "crf", "dom"])
    p("--crf-alignment-dir", type=Path, help="Directory with CRFalign alignments")
    p("--kalign-bin", default="kalign", help="Kalign executable")
    p("--seed", type=int, help="Random seed for shuffling template hits")
    p("--offset", type=int, default=0, help="Residue index offset")
    return parser


def cli(argv: List[str] | None = None) -> None:  # pragma: no cover
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    setup_logging("features.log", mode="a")

    build_input_features(
        fasta_path=args.fasta,
        alignment_paths=args.alignments,
        template_path=args.template,
        output_path=args.output,
        pdb_mmcif_dir=args.pdb_mmcif_dir,
        pdb_obsolete_path=args.pdb_obsolete,
        max_template_date=args.max_template_date,
        max_template_hits=args.max_template_hits,
        template_mode=args.template_mode,
        crf_alignment_dir=args.crf_alignment_dir,
        kalign_bin=args.kalign_bin,
        seed=args.seed,
        offset=args.offset,
    )


if __name__ == "__main__":  # pragma: no cover
    cli()
