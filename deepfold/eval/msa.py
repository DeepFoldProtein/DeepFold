import string

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from scipy.spatial.distance import pdist, squareform

from deepfold.eval.utils import read_iter

RESTYPES = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V", "X", "-"]

RESTYPE_MAPPING = {r: i for i, r in enumerate(RESTYPES)}
RESTYPE_MAPPING["B"] = RESTYPES.index("D")
RESTYPE_MAPPING["J"] = RESTYPES.index("X")
RESTYPE_MAPPING["O"] = RESTYPES.index("X")
RESTYPE_MAPPING["U"] = RESTYPES.index("C")
RESTYPE_MAPPING["Z"] = RESTYPES.index("E")


def compute_neff_v1(
    msa: np.ndarray,
    cutoff: float = 0.62,
    eps: float = 1e-6,
) -> float:
    # assert cutoff > 0.0
    y = pdist(msa, metric="hamming")
    d = squareform(y)
    w = d > cutoff
    neff = np.sum(1.0 / (w.sum(axis=0) + eps))
    return neff


def compute_neff_v2(
    msa: np.ndarray,
    cutoff: float = 0.62,
    eps: float = 1e-10,
) -> float:
    theta = 1.0 - cutoff
    assert theta > 0.0
    y = pdist(msa, metric="hamming")
    d = squareform(y)
    w = 1.0 / (1.0 + np.sum(d < theta, 0))
    neff = np.log2(eps + np.sum(w))
    return neff


def create_msa_array(
    a3m_strings: list[str],
    num_sym: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if num_sym is None:
        num_sym = [1] * len(a3m_strings)
    assert len(a3m_strings) == len(num_sym)
    msa_arrays = []
    asym_id = []

    table = str.maketrans("", "", string.ascii_lowercase)
    aid = 1
    for idx, a3m_string in enumerate(a3m_strings):
        lines = [s.translate(table) for _, s in read_iter(a3m_string)]
        num_alignments = len(lines)
        query_length = len(lines[0])
        msa = np.full(
            (num_alignments, query_length),
            RESTYPE_MAPPING["-"],
            dtype=np.int8,
        )
        for i, line in enumerate(lines):
            msa[i, :] = [RESTYPE_MAPPING[r] for r in line]
        for _ in range(num_sym[idx]):
            asym_id.append(np.full(len(lines[0]), aid, dtype=np.int64))
            msa_arrays.append(msa)
            aid += 1

    return np.column_stack(msa_arrays), np.concatenate(asym_id)


def plot_msa_depth(
    msa: np.ndarray,
    asym_id: np.ndarray,
    sort_lines: bool = True,
    ax: Axes | None = None,
) -> Axes:
    """
    Visualizes the multiple sequence alignment (MSA) with sequence identity to the query.

    Args:
        msa (np.ndarray): A 2-diemnsional array of the multiple sequence alignment.
        asym_id (np.ndarray): A 1-dimensional array indicating the chain or asymmetric unit
            for each residue in the query sequence. Used to delineate chains in the plot.
        sort_lines (bool, optional): Whether to sort the alignment lines based on
            their average sequence identity to the query. Defaults to True.

    Returns:
        matplotlib.axes.Axes: The matplotlib Axes object.
    """
    if ax is None:
        ax = plt.gca()

    # Extract the query sequence from the MSA. It's assumed to be the first sequence.
    query_sequence = msa[0]

    # Determine the boundaries of different asymmetric units (chains) if 'asym_id' is provided.
    chain_lengths = [0]
    current_chain_id = asym_id[0]
    for chain_id in asym_id:
        if chain_id == current_chain_id:
            chain_lengths[-1] += 1
        else:
            chain_lengths.append(1)
            current_chain_id = chain_id

    # Calculate the cumulative lengths to get the start and end indices of each chain.
    chain_boundaries = np.cumsum([0] + chain_lengths)

    # Create a boolean mask indicating non-gap positions (amino acid code != 21).
    is_non_gap = msa != RESTYPE_MAPPING["-"]

    # Create a boolean mask indicating positions where the aligned sequence matches the query.
    is_identical = msa == query_sequence

    # Calculate the fraction of non-gap residues for each alignment within each chain.
    chain_non_gap_fraction = np.stack(
        [is_non_gap[:, chain_boundaries[i] : chain_boundaries[i + 1]].max(axis=-1) for i in range(len(chain_lengths))],
        axis=-1,
    )

    lines = []
    num_lines_per_group = []

    # Iterate through unique patterns of gap distribution across chains.
    for gap_pattern in np.unique(chain_non_gap_fraction, axis=0):
        # Find the indices of alignments with the current gap pattern.
        indices = np.where((chain_non_gap_fraction == gap_pattern).all(axis=-1))[0]
        identical_subsequences = is_identical[indices]
        non_gap_subsequences = is_non_gap[indices]

        # Calculate the average sequence identity to the query for each alignment,
        # normalizing by the number of non-gap residues in each chain.
        sequence_identity = np.stack(
            [identical_subsequences[:, chain_boundaries[i] : chain_boundaries[i + 1]].mean(axis=-1) for i in range(len(chain_lengths))],
            axis=-1,
        ).sum(axis=-1) / (gap_pattern.sum(axis=-1) + 1e-8)

        # Convert the boolean non-gap mask to float for multiplication.
        non_gaps_float = non_gap_subsequences.astype(float)
        non_gaps_float[non_gaps_float == 0] = np.nan  # Mark gaps as NaN for visualization.

        # Sort the lines based on sequence identity if sort_lines is True.
        if sort_lines:
            sorted_indices = sequence_identity.argsort()
            lines_to_add = non_gaps_float[sorted_indices] * sequence_identity[sorted_indices, None]
        else:
            lines_to_add = non_gaps_float[::-1] * sequence_identity[::-1, None]

        num_lines_per_group.append(len(lines_to_add))
        lines.append(lines_to_add)

    # Concatenate the lines from different gap pattern groups.
    num_lines_cumulative = np.cumsum(np.append(0, num_lines_per_group))
    all_lines = np.concatenate(lines, axis=0)

    # Create the plot.
    if all_lines.size > 0:
        im = ax.imshow(
            all_lines,
            interpolation="nearest",
            aspect="auto",
            cmap="rainbow_r",
            vmin=0,
            vmax=1,
            origin="lower",
            extent=(0, all_lines.shape[1], 0, all_lines.shape[0]),
        )

        # Add vertical lines to separate different asymmetric units (chains).
        for boundary in chain_boundaries[1:-1]:
            # ax.plot([boundary, boundary], [0, all_lines.shape[0]], color="black")
            ax.axvline(x=boundary - 0.5, color="black")

        # Add horizontal lines to separate groups of alignments with different gap patterns.
        for boundary in num_lines_cumulative[1:-1]:
            # ax.plot([0, all_lines.shape[1]], [boundary, boundary], color="black")
            ax.axhline(y=boundary - 0.5, color="black")

        # Coverage plot
        coverage = (~np.isnan(all_lines)).sum(axis=0)
        # Adjust vertical position as needed
        ax.plot(coverage, color="black", linewidth=1)

        ax.set_xlim(0, all_lines.shape[1])
        # Adjust y-limit to accommodate coverage plot
        ax.set_ylim(0, all_lines.shape[0] + 3)
        ax.figure.colorbar(im)  # , label="Sequence identity to query")
    else:
        ax.set_xlim(0, msa.shape[1])
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.5, 0.5, "No aligned sequences to display.", ha="center", va="center")

    # ax.set_title("Sequence coverage")
    ax.set_xlabel("Positions")
    ax.set_ylabel("Sequences")

    return ax
