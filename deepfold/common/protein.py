# Copyright 2024 DeepFold Team
# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Protein data type."""


import dataclasses
import io
from typing import Any, Iterable, List, Mapping, Optional, Sequence

import numpy as np
from Bio.PDB import PDBParser

from deepfold.common import residue_constants as rc

FeatureDict = Mapping[str, np.ndarray]
ModelOutput = Mapping[str, Any]  # Is a nested dict.
PICO_TO_ANGSTROM = 0.01

PDB_CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
PDB_MAX_CHAINS = len(PDB_CHAIN_IDS)
assert PDB_MAX_CHAINS == 62


@dataclasses.dataclass(frozen=True)
class Protein:
    """Protein structure representation."""

    # Cartesian coordinates of atoms in angstroms. The atom types correspond to
    # residue_constants.atom_types, i.e. the first three are N, CA, CB.
    atom_positions: np.ndarray  # [num_res, num_atom_type, 3]

    # Amino-acid type for each residue represented as an integer between 0 and
    # 20, where 20 is 'X'.
    aatype: np.ndarray  # [num_res]

    # Binary float mask to indicate presence of a particular atom. 1.0 if an atom
    # is present and 0.0 if not. This should be used for loss masking.
    atom_mask: np.ndarray  # [num_res, num_atom_type]

    # Residue index as used in PDB. It is not necessarily continuous or 0-indexed.
    residue_index: np.ndarray  # [num_res]

    # B-factors, or temperature factors, of each residue (in sq. angstroms units),
    # representing the displacement of the residue from its ground truth mean
    # value.
    b_factors: np.ndarray  # [num_res, num_atom_type]

    # Chain indices for multi-chain predictions
    chain_index: Optional[np.ndarray] = None

    # Optional remark about the protein. Included as a comment in output PDB
    # files
    remark: Optional[str] = None

    # Templates used to generate this protein (prediction-only)
    parents: Optional[Sequence[str]] = None

    # Chain corresponding to each parent
    parents_chain_index: Optional[Sequence[int]] = None

    def __post_init__(self):
        if self.chain_index is not None and len(np.unique(self.chain_index)) > PDB_MAX_CHAINS:
            raise ValueError(f"Cannot build an instance with more than {PDB_MAX_CHAINS} " "chains because these cannot be written to PDB format")


def from_pdb_string(pdb_str: str, chain_id: Optional[str] = None) -> Protein:
    """Takes a PDB string and constructs a Protein object.

    WARNING: All non-standard residue types will be converted into UNK. All
      non-standard atoms will be ignored.

    Args:
      pdb_str: The contents of the pdb file
      chain_id: If None, then the whole pdb file is parsed. If chain_id is specified (e.g. A), then only that chain
        is parsed.

    Returns:
      A new `Protein` parsed from the pdb contents.
    """
    pdb_fh = io.StringIO(pdb_str)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("none", pdb_fh)
    models = list(structure.get_models())
    if len(models) != 1:
        raise ValueError(f"Only single model PDBs are supported. Found {len(models)} models.")
    model = models[0]

    atom_positions = []
    aatype = []
    atom_mask = []
    residue_index = []
    chain_ids = []
    b_factors = []

    for chain in model:
        if chain_id is not None and chain.id != chain_id:
            continue

        for res in chain:
            if res.id[2] != " ":
                raise ValueError(f"PDB contains an insertion code at chain {chain.id} and residue " f"index {res.id[1]}. These are not supported.")
            res_shortname = rc.restype_3to1.get(res.resname, "X")
            restype_idx = rc.restype_order.get(res_shortname, rc.restype_num)
            pos = np.zeros((rc.atom_type_num, 3))
            mask = np.zeros((rc.atom_type_num,))
            res_b_factors = np.zeros((rc.atom_type_num,))
            for atom in res:
                if atom.name not in rc.atom_types:
                    continue
                pos[rc.atom_order[atom.name]] = atom.coord
                mask[rc.atom_order[atom.name]] = 1.0
                res_b_factors[rc.atom_order[atom.name]] = atom.bfactor
            if np.sum(mask) < 0.5:
                # If no known atom positions are reported for the residue then skip it.
                continue

            aatype.append(restype_idx)
            atom_positions.append(pos)
            atom_mask.append(mask)
            residue_index.append(res.id[1])
            chain_ids.append(chain.id)
            b_factors.append(res_b_factors)

    parents = None
    parents_chain_index = None
    if "PARENT" in pdb_str:
        parents = []
        parents_chain_index = []
        chain_count = 0
        for l in pdb_str.split("\n"):
            if "PARENT" in l:
                if not "N/A" in l:
                    parent_names = l.split()[1:]
                    parents.extend(parent_names)
                    parents_chain_index.extend([chain_count for _ in parent_names])
                chain_count += 1

    # unique_chain_ids = np.unique(chain_ids)
    chain_id_mapping = {cid: n for n, cid in enumerate(PDB_CHAIN_IDS)}
    chain_index = np.array([chain_id_mapping[cid] for cid in chain_ids])

    return Protein(
        atom_positions=np.array(atom_positions),
        atom_mask=np.array(atom_mask),
        aatype=np.array(aatype),
        residue_index=np.array(residue_index),
        chain_index=chain_index,
        b_factors=np.array(b_factors),
        parents=parents,
        parents_chain_index=parents_chain_index,
    )


def _chain_end(atom_index, end_resname, chain_name, residue_index) -> str:
    chain_end = "TER"
    return f"{chain_end:<6}{atom_index:>5}      {end_resname:>3} " f"{chain_name:>1}{residue_index:>4}"


def get_pdb_headers(prot: Protein, chain_id: int = 0) -> Sequence[str]:
    pdb_headers = []

    remark = prot.remark
    if remark is not None:
        pdb_headers.append(f"REMARK {remark}")

    parents = prot.parents
    parents_chain_index = prot.parents_chain_index
    if parents_chain_index is not None:
        parents = [p for i, p in zip(parents_chain_index, parents) if i == chain_id]

    if parents is None or len(parents) == 0:
        parents = ["N/A"]

    pdb_headers.append(f"PARENT {' '.join(parents)}")

    return pdb_headers


def add_pdb_headers(prot: Protein, pdb_str: str) -> str:
    """Add pdb headers to an existing PDB string. Useful during multi-chain
    recycling
    """
    out_pdb_lines = []
    lines = pdb_str.split("\n")

    remark = prot.remark
    if remark is not None:
        out_pdb_lines.append(f"REMARK {remark}")

    parents_per_chain = None
    if prot.parents is not None and len(prot.parents) > 0:
        parents_per_chain = []
        if prot.parents_chain_index is not None:
            cur_chain = prot.parents_chain_index[0]
            parent_dict = {}
            for p, i in zip(prot.parents, prot.parents_chain_index):
                parent_dict.setdefault(str(i), [])
                parent_dict[str(i)].append(p)

            max_idx = max([int(chain_idx) for chain_idx in parent_dict])
            for i in range(max_idx + 1):
                chain_parents = parent_dict.get(str(i), ["N/A"])
                parents_per_chain.append(chain_parents)
        else:
            parents_per_chain.append(prot.parents)
    else:
        parents_per_chain = [["N/A"]]

    make_parent_line = lambda p: f"PARENT {' '.join(p)}"

    out_pdb_lines.append(make_parent_line(parents_per_chain[0]))

    chain_counter = 0
    for i, l in enumerate(lines):
        if "PARENT" not in l and "REMARK" not in l:
            out_pdb_lines.append(l)
        if "TER" in l and not "END" in lines[i + 1]:
            chain_counter += 1
            if not chain_counter >= len(parents_per_chain):
                chain_parents = parents_per_chain[chain_counter]
            else:
                chain_parents = ["N/A"]

            out_pdb_lines.append(make_parent_line(chain_parents))

    return "\n".join(out_pdb_lines)


def to_pdb(prots: Protein | Iterable[Protein]) -> str:
    """Convert `Protein` instances to a PDB string."""

    if isinstance(prots, Protein):
        prots = [prots]
    elif isinstance(prots, Iterable):
        prots = list(prots)
    else:
        raise ValueError("Unknown type")

    pdb_lines = []

    for model_id, prot in enumerate(prots, start=1):
        pdb_lines.extend(_single_protein_to_pdb(prot, model_id=model_id))

    pdb_lines.append("END")

    # Pad all lines to 80 characters
    pdb_lines = [line.ljust(80) for line in pdb_lines]
    return "\n".join(pdb_lines) + "\n"  # Add terminating newline.


def _single_protein_to_pdb(prot: Protein, model_id: int = 1) -> List[str]:
    """Converts a `Protein` instance to a PDB string.

    Args:
      prot: The protein to convert to PDB.

    Returns:
      PDB string.
    """
    restypes = rc.restypes + ["X"]
    res_1to3 = lambda r: rc.restype_1to3.get(restypes[r], "UNK")
    atom_types = rc.atom_types

    pdb_lines = []

    atom_mask = prot.atom_mask
    aatype = prot.aatype
    atom_positions = prot.atom_positions
    residue_index = prot.residue_index.astype(np.int32)
    b_factors = prot.b_factors
    chain_index = prot.chain_index.astype(np.int32)

    if np.any(aatype > rc.restype_num):
        raise ValueError("Invalid aatypes.")

    # Construct a mapping from chain integer indices to chain ID strings.
    chain_ids = {}
    for i in np.unique(chain_index):  # np.unique gives sorted output.
        if i >= PDB_MAX_CHAINS:
            raise ValueError(f"The PDB format supports at most {PDB_MAX_CHAINS} chains.")
        chain_ids[i] = PDB_CHAIN_IDS[i]

    headers = get_pdb_headers(prot)
    if len(headers) > 0:
        pdb_lines.extend(headers)

    pdb_lines.append(f"MODEL     {model_id:4d}")
    n = aatype.shape[0]
    atom_index = 1
    last_chain_index = chain_index[0]
    prev_chain_index = 0
    chain_tags = PDB_CHAIN_IDS

    # Add all atom sites.
    for i in range(aatype.shape[0]):
        # Close the previous chain if in a multichain PDB.
        # if last_chain_index != chain_index[i]:
        #     pdb_lines.append(
        #         _chain_end(
        #             atom_index,
        #             res_1to3(aatype[i - 1]),
        #             chain_ids[chain_index[i - 1]],
        #             residue_index[i - 1],
        #         )
        #     )
        #     last_chain_index = chain_index[i]
        #     atom_index += 1  # Atom index increases at the TER symbol.

        res_name_3 = res_1to3(aatype[i])
        for atom_name, pos, mask, b_factor in zip(atom_types, atom_positions[i], atom_mask[i], b_factors[i]):
            if mask < 0.5:
                continue

            record_type = "ATOM"
            name = atom_name if len(atom_name) == 4 else f" {atom_name}"
            alt_loc = ""
            insertion_code = ""
            occupancy = 1.00
            element = atom_name[0]  # Protein supports only C, N, O, S, this works.
            charge = ""

            chain_tag = "A"
            if chain_index is not None:
                chain_tag = chain_tags[chain_index[i]]

            # PDB is a columnar format, every space matters here!
            atom_line = (
                f"{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}"
                f"{res_name_3:>3} {chain_tag:>1}"
                f"{residue_index[i]:>4}{insertion_code:>1}   "
                f"{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}"
                f"{occupancy:>6.2f}{b_factor:>6.2f}          "
                f"{element:>2}{charge:>2}"
            )
            pdb_lines.append(atom_line)
            atom_index += 1

        should_terminate = i == n - 1
        if chain_index is not None:
            if i != n - 1 and chain_index[i + 1] != prev_chain_index:
                should_terminate = True
                prev_chain_index = chain_index[i + 1]

        if should_terminate:
            # Close the chain.
            chain_end = "TER"
            chain_termination_line = f"{chain_end:<6}{atom_index:>5}      " f"{res_1to3(aatype[i]):>3} " f"{chain_tag:>1}{residue_index[i]:>4}"
            pdb_lines.append(chain_termination_line)
            atom_index += 1

            if i != n - 1:
                # "prev" is a misnomer here. This happens at the beginning of
                # each new chain.
                pdb_lines.extend(get_pdb_headers(prot, prev_chain_index))

    pdb_lines.append("ENDMDL")

    return pdb_lines


def ideal_atom_mask(prot: Protein) -> np.ndarray:
    """Computes an ideal atom mask.

    `Protein.atom_mask` typically is defined according to the atoms that are
    reported in the PDB. This function computes a mask according to heavy atoms
    that should be present in the given sequence of amino acids.

    Args:
      prot: `Protein` whose fields are `numpy.ndarray` objects.

    Returns:
      An ideal atom mask.
    """
    return rc.STANDARD_ATOM_MASK[prot.aatype]


def from_prediction(
    processed_features: FeatureDict,
    result: ModelOutput,
    b_factors: Optional[np.ndarray] = None,
    remove_leading_feature_dimension: bool = False,
    is_trajectory: bool = False,
    remark: Optional[str] = None,
    parents: Optional[Sequence[str]] = None,
    parents_chain_index: Optional[Sequence[int]] = None,
) -> Protein | List[Protein]:
    """Assembles a protein from a prediction.

    Args:
      processed_features: Dictionary holding model inputs.
      result: Dictionary holding model outputs.
      b_factors: (Optional) B-factors to use for the protein.
      remove_leading_feature_dimension: Whether to remove the leading dimension of the feature values.
      remark: (Optional) Remark about the prediction
      parents: (Optional) List of template names
    Returns:
      A protein instance.
    """

    def _maybe_remove_leading_dim(arr: np.ndarray) -> np.ndarray:
        return arr[0] if remove_leading_feature_dimension else arr

    if "asym_id" in processed_features:
        chain_index = _maybe_remove_leading_dim(processed_features["asym_id"].astype(np.int32)) - 1
    else:
        chain_index = np.zeros_like(_maybe_remove_leading_dim(processed_features["aatype"]), dtype=np.int32)

    if b_factors is None:
        b_factors = np.zeros_like(result["final_atom_mask"])
    if b_factors.ndim == 1:
        b_factors = np.repeat(b_factors[..., None], rc.atom_type_num, axis=-1)

    if is_trajectory:
        proteins = []
        for i in range(result["trajectory"].shape[-1]):
            prot = Protein(
                aatype=_maybe_remove_leading_dim(processed_features["aatype"]),
                atom_positions=result["trajectory"][..., i],
                atom_mask=result["final_atom_mask"],
                residue_index=_maybe_remove_leading_dim(processed_features["residue_index"]) + 1,
                b_factors=b_factors,
                chain_index=chain_index,
                remark=f"recycle={i} {remark}",
                parents=parents,
                parents_chain_index=parents_chain_index,
            )
            proteins.append(prot)
        return proteins
    else:
        return Protein(
            aatype=_maybe_remove_leading_dim(processed_features["aatype"]),
            atom_positions=result["final_atom_positions"],
            atom_mask=result["final_atom_mask"],
            residue_index=_maybe_remove_leading_dim(processed_features["residue_index"]) + 1,
            b_factors=b_factors,
            chain_index=chain_index,
            remark=remark,
            parents=parents,
            parents_chain_index=parents_chain_index,
        )


def from_relaxation(
    relaxed_pdb_str: str,
    residue_index: np.ndarray | None = None,
    chain_index: np.ndarray | None = None,
    b_factors: np.ndarray | None = None,
) -> Protein:
    """Amber relaxation procedure renames residue index starting from 1.
    Since we may ahve cropped domains, we must fix residue indices with correct ones.

    Args:
        relaxed_pdb_str: a protein indices/
        residue_index: residue indcies.

    Returns:
        PDB strings.

    """
    relaxed_prot = from_pdb_string(relaxed_pdb_str)
    return Protein(
        aatype=relaxed_prot.aatype,
        atom_positions=relaxed_prot.atom_positions,
        atom_mask=relaxed_prot.atom_mask,
        residue_index=relaxed_prot.residue_index if residue_index is None else residue_index,
        chain_index=relaxed_prot.chain_index if chain_index is None else chain_index,
        b_factors=relaxed_prot.b_factors if b_factors is not None else b_factors,
    )
