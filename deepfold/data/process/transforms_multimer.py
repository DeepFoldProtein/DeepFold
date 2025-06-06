import logging
from typing import Dict, Sequence

import torch
import torch.nn.functional as F

from deepfold.common import residue_constants as rc
from deepfold.config import NUM_RES
from deepfold.data.process.transforms import curry1
from deepfold.utils.tensor_utils import masked_mean

logger = logging.getLogger(__name__)


def gumbel_noise(
    shape: Sequence[int],
    device: torch.device,
    eps: float = 1e-6,
    generator: torch.Generator = None,
) -> torch.Tensor:
    """Generate Gumbel noise of given shape."""
    uniform_noise = torch.rand(shape, dtype=torch.float32, device=device, generator=generator)
    gumbel = -torch.log(-torch.log(uniform_noise + eps) + eps)
    return gumbel


def gumbel_max_sample(
    logits: torch.Tensor,
    generator: torch.Generator = None,
) -> torch.Tensor:
    """Samples from a probabiliy distribution given by 'logits'."""
    z = gumbel_noise(logits.shape, logits.device, generator=generator)
    return F.one_hot(torch.argmax(logits + z, dim=-1), num_classes=logits.size(-1))


def gumbel_argsort_sample_idx(
    logits: torch.Tensor,
    generator: torch.Generator = None,
) -> torch.Tensor:
    """Samples with replacement from a distribution given by `logits`."""
    z = gumbel_noise(logits.shape, logits.device, generator=generator)
    return torch.argsort(logits + z, dim=-1, descending=True)


@curry1
def make_masked_msa(
    batch: dict,
    masked_msa_profile_prob: float,
    masked_msa_same_prob: float,
    masked_msa_uniform_prob: float,
    masked_msa_replace_fraction: float,
    seed: int | None = None,
    eps: float = 1e-6,
):
    """Create data for BERT on raw MSA."""
    # Add a random amino acid uniformly.
    random_aa = torch.Tensor([0.05] * 20 + [0.0, 0.0], device=batch["msa"].device)

    categorical_probs = (
        masked_msa_uniform_prob * random_aa + masked_msa_profile_prob * batch["msa_profile"] + masked_msa_same_prob * F.one_hot(batch["msa"], 22)
    )

    # Put all remaining probability on [MASK] which is a new column.
    mask_prob = 1.0 - masked_msa_profile_prob - masked_msa_same_prob - masked_msa_uniform_prob

    categorical_probs = F.pad(categorical_probs, [0, 1], value=mask_prob)

    sh = batch["msa"].shape
    mask_position = torch.rand(sh, device=batch["msa"].device) < masked_msa_replace_fraction
    mask_position *= batch["msa_mask"].to(mask_position.dtype)

    logits = torch.log(categorical_probs + eps)

    g = None
    if seed is not None:
        g = torch.Generator(device=batch["msa"].device)
        g.manual_seed(seed)

    bert_msa = gumbel_max_sample(logits, generator=g)

    bert_msa = torch.where(mask_position, torch.argmax(bert_msa, dim=-1), batch["msa"])
    bert_msa *= batch["msa_mask"].to(bert_msa.dtype)

    # Mix real and masked MSA.
    if "bert_mask" in batch:
        batch["bert_mask"] *= mask_position.to(torch.float32)
    else:
        batch["bert_mask"] = mask_position.to(torch.float32)
    batch["true_msa"] = batch["msa"]
    batch["msa"] = bert_msa

    return batch


@curry1
def nearest_neighbor_clusters(
    batch: dict,
    gap_agreement_weight: float,  # = 0.0,
):
    """Assign each extra MSA sequence to its nearest neighbor in sampled MSA."""
    device = batch["msa_mask"].device

    # Determine how much weight we assign to each agreement.
    # In theory, we could use a full blosum matrix here,
    # but right now let's just down-weight gap agreement because it could be spurious.
    # Never put weight on agreeing on BERT mask.

    weights = torch.Tensor(
        [1.0] * 21 + [gap_agreement_weight] + [0.0],
        device=device,
    )

    msa_mask = batch["msa_mask"]
    msa_one_hot = F.one_hot(batch["msa"], 23)

    extra_mask = batch["extra_msa_mask"]
    extra_one_hot = F.one_hot(batch["extra_msa"], 23)

    msa_one_hot_masked = msa_mask[:, :, None] * msa_one_hot
    extra_one_hot_masked = extra_mask[:, :, None] * extra_one_hot

    agreement = torch.einsum("mrc, nrc->nm", extra_one_hot_masked, weights * msa_one_hot_masked)

    cluster_assignment = F.softmax(1e3 * agreement, dim=0)
    cluster_assignment *= torch.einsum("mr, nr->mn", msa_mask, extra_mask)

    cluster_count = torch.sum(cluster_assignment, dim=-1)
    cluster_count += 1.0  # We always include the sequence itself.

    msa_sum = torch.einsum("nm, mrc->nrc", cluster_assignment, extra_one_hot_masked)
    msa_sum += msa_one_hot_masked

    cluster_profile = msa_sum / cluster_count[:, None, None]

    extra_deletion_matrix = batch["extra_deletion_matrix"]
    deletion_matrix = batch["deletion_matrix"]

    del_sum = torch.einsum("nm, mc->nc", cluster_assignment, extra_mask * extra_deletion_matrix)
    del_sum += deletion_matrix  # Original sequence.
    cluster_deletion_mean = del_sum / cluster_count[:, None]

    batch["cluster_profile"] = cluster_profile
    batch["cluster_deletion_mean"] = cluster_deletion_mean

    return batch


def create_target_feat(batch: dict):
    """Create the target features"""
    batch["target_feat"] = F.one_hot(batch["aatype"], num_classes=21).to(torch.float32)
    return batch


def create_msa_feat(batch: dict):
    """Create and concatenate MSA features."""
    device = batch["msa"]
    msa_1hot = F.one_hot(batch["msa"], 23)
    deletion_matrix = batch["deletion_matrix"]
    has_deletion = torch.clamp(deletion_matrix, min=0.0, max=1.0)[..., None]
    pi = torch.acos(torch.zeros(1, device=deletion_matrix.device)) * 2
    deletion_value = (torch.atan(deletion_matrix / 3.0) * (2.0 / pi))[..., None]

    deletion_mean_value = (torch.atan(batch["cluster_deletion_mean"] / 3.0) * (2.0 / pi))[..., None]

    msa_feat = torch.cat(
        [msa_1hot, has_deletion, deletion_value, batch["cluster_profile"], deletion_mean_value],
        dim=-1,
    )

    batch["msa_feat"] = msa_feat

    return batch


def build_extra_msa_feat(batch: dict):
    """Expand extra_msa into 1hot and concat with other extra msa features.

    We do this as late as possible as the one_hot extra msa can be very large.

    Args:
        batch: a dictionary with the following keys:
         * 'extra_msa': [num_seq, num_res] MSA that wasn't selected as a cluster
             centre. Note - This isn't one-hotted.
         * 'extra_deletion_matrix': [num_seq, num_res] Number of deletions at given
                position.
        num_extra_msa: Number of extra msa to use.

    Returns:
        Concatenated tensor of extra MSA features.
    """
    # 23 = 20 amino acids + 'X' for unknown + gap + bert mask
    extra_msa = batch["extra_msa"]
    deletion_matrix = batch["extra_deletion_matrix"]
    msa_1hot = F.one_hot(extra_msa, 23)
    has_deletion = torch.clamp(deletion_matrix, min=0.0, max=1.0)[..., None]
    pi = torch.acos(torch.zeros(1, device=deletion_matrix.device)) * 2
    deletion_value = (torch.atan(deletion_matrix / 3.0) * (2.0 / pi))[..., None]
    extra_msa_mask = batch["extra_msa_mask"]
    catted = torch.cat([msa_1hot, has_deletion, deletion_value], dim=-1)

    return catted


@curry1
def sample_msa(
    batch: dict,
    max_msa_clusters: int,
    max_extra_msa_seq: int,
    seed: int | None = None,
):
    """Sample MSA randomly, remaining sequences are stored as `extra_*`.

    Args:
        batch: batch to sample msa from.
        max_seq: number of sequences to sample.
    Returns:
        Protein with sampled msa.
    """
    assert seed is not None
    g = torch.Generator(device=batch["msa"].device)
    g.manual_seed(seed)

    mask = torch.clamp(torch.sum(batch["msa_mask"], dim=-1), 0.0, 1.0)
    if "msa_weight" in batch:
        assert mask.shape == batch["msa_weight"].shape
        prob = batch["msa_weight"] * mask
    else:
        prob = mask
    prob[0] = 0.0
    prob /= prob.sum() + 1e-6

    picked = torch.multinomial(prob, prob.numel() - 1, replacement=False, generator=g)
    dims = [*prob.shape[:-1], 1]
    index_order = torch.cat([torch.zeros(dims, dtype=torch.long, device=prob.device), picked])

    sel_idx = index_order[:max_msa_clusters]
    extra_idx = index_order[max_msa_clusters:][:max_extra_msa_seq]

    for k in ["msa", "deletion_matrix", "msa_mask", "bert_mask"]:
        if k in batch:
            batch["extra_" + k] = batch[k][extra_idx]
            batch[k] = batch[k][sel_idx]

    return batch


def make_msa_profile(batch: dict):
    """Compute the MSA profile."""
    batch["msa_profile"] = masked_mean(
        batch["msa_mask"][:, :, None],
        F.one_hot(batch["msa"], num_classes=22),
        dim=-3,
        eps=1e-10,
    )
    return batch


def _randint(
    lower: int,
    upper: int,
    generator: torch.Generator,
    device: torch.device,
) -> int:
    return int(
        torch.randint(
            lower,
            upper + 1,
            (1,),
            generator=generator,
            device=device,
        ).item()
    )


def get_interface_residues(
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    asym_id: torch.Tensor,
    interface_threshold: float,
) -> torch.Tensor:
    displacements = all_atom_positions[..., :, None, :, :] - all_atom_positions[..., None, :, :, :]
    pairwise_distances = torch.sqrt(torch.sum(displacements**2, dim=-1))

    chain_mask_2d = (asym_id[..., :, None, :] != asym_id[..., :, :, None]).float()
    pair_mask = all_atom_mask[..., :, None, :] * all_atom_mask[..., None, :, :]
    mask = (chain_mask_2d[..., None] * pair_mask).bool()

    minimum_distance_per_residues, _ = torch.where(mask, pairwise_distances, torch.inf).min(dim=-1)

    valid_interfaces = torch.sum((minimum_distance_per_residues < interface_threshold).float(), dim=-1)
    interface_residues_indices = torch.nonzero(valid_interfaces, as_tuple=True)[0]

    return interface_residues_indices


def get_spatial_crop_idx(
    protein: Dict[str, torch.Tensor],
    sequence_crop_size: int,
    interface_threshold: float,
    generator: torch.Generator,
) -> torch.Tensor:

    positions = protein["all_atom_positions"]
    atom_mask = protein["all_atom_mask"]
    asym_id = protein["asym_id"]

    interface_residues = get_interface_residues(
        all_atom_positions=positions,
        all_atom_mask=atom_mask,
        asym_id=asym_id,
        interface_threshold=interface_threshold,
    )

    if not torch.any(interface_residues):
        return get_contiguous_crop_idx(protein, sequence_crop_size, generator)

    target_res_idx = _randint(0, interface_residues.size(-1) - 1, generator, positions.device)

    target_res = interface_residues[target_res_idx]

    ca_idx = rc.atom_order["CA"]
    ca_positions = positions[..., ca_idx, :]
    ca_mask = atom_mask[..., ca_idx].bool()

    coord_diff = ca_positions[..., :, None, :] - ca_positions[..., None, :, :]
    ca_pairwise_dists = torch.sqrt(torch.sum(coord_diff**2, dim=-1))

    to_target_distances = ca_pairwise_dists[target_res]
    break_tie = torch.arange(0, to_target_distances.shape[-1], device=positions.device).float() * 1e-3
    to_target_distances = torch.where(ca_mask, to_target_distances, torch.inf) + break_tie

    ret = torch.argsort(to_target_distances)[:sequence_crop_size]
    return ret.sort().values


def get_contiguous_crop_idx(
    protein: Dict[str, torch.Tensor],
    sequence_crop_size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    unique_asym_ids, chain_idxs, chain_lens = protein["asym_id"].unique(
        dim=-1,
        return_inverse=True,
        return_counts=True,
    )

    shuffle_idx = torch.randperm(
        chain_lens.shape[-1],
        device=chain_lens.device,
        generator=generator,
    )

    _, idx_sorted = torch.sort(chain_idxs, stable=True)
    cum_sum = chain_lens.cumsum(dim=0)
    cum_sum = torch.cat((torch.tensor([0]), cum_sum[:-1]), dim=0)
    asym_offsets = idx_sorted[cum_sum]

    num_budget = sequence_crop_size
    num_remaining = int(protein["seq_length"])

    crop_idxs = []
    for idx in shuffle_idx:
        chain_len = int(chain_lens[idx])
        num_remaining -= chain_len

        crop_size_max = min(num_budget, chain_len)
        crop_size_min = min(chain_len, max(0, num_budget - num_remaining))
        chain_crop_size = _randint(
            lower=crop_size_min,
            upper=crop_size_max,
            generator=generator,
            device=chain_lens.device,
        )

        num_budget -= chain_crop_size

        chain_start = _randint(
            lower=0,
            upper=chain_len - chain_crop_size,
            generator=generator,
            device=chain_lens.device,
        )

        asym_offset = asym_offsets[idx]
        crop_idxs.append(torch.arange(asym_offset + chain_start, asym_offset + chain_start + chain_crop_size))

    return torch.concat(crop_idxs).sort().values


@curry1
def random_crop_and_template_subsampling(
    protein: Dict[str, torch.Tensor],
    sequence_crop_size: int,
    max_templates: int,
    feature_schema_shapes: Dict[str, tuple],
    spatial_crop_prob: float,
    interface_threshold: float,
    subsample_templates: bool,
    seed: int | None = None,
) -> Dict[str, torch.Tensor]:
    """Crop randomly to `crop_size`, or keep as is if shorter than that."""
    # We want each ensemble to be cropped the same way
    g = None
    if seed is not None:
        g = torch.Generator(device=protein["seq_length"].device)
        g.manual_seed(seed)

    use_spatial_crop = torch.rand((1,), device=protein["seq_length"].device, generator=g) < spatial_crop_prob

    num_res = protein["aatype"].shape[0]
    if num_res <= sequence_crop_size:
        crop_idxs = torch.arange(num_res)
    elif use_spatial_crop:
        crop_idxs = get_spatial_crop_idx(protein, sequence_crop_size, interface_threshold, g)
    else:
        crop_idxs = get_contiguous_crop_idx(protein, sequence_crop_size, g)

    num_templates = protein["template_aatype"].shape[0]

    # No need to subsample templates if there aren't any
    subsample_templates = subsample_templates and num_templates

    if subsample_templates:
        templates_crop_start = _randint(
            lower=0,
            upper=num_templates,
            generator=g,
            device=protein["seq_length"].device,
        )
        templates_select_indices = torch.randperm(
            num_templates,
            device=protein["seq_length"].device,
            generator=g,
        )
    else:
        templates_crop_start = 0

    num_res_crop_size = min(int(protein["seq_length"]), sequence_crop_size)
    num_templates_crop_size = min(num_templates - templates_crop_start, max_templates)

    for k, v in protein.items():
        if k not in feature_schema_shapes or ("template" not in k and NUM_RES not in feature_schema_shapes[k]):
            continue

        # randomly permute the templates before cropping them.
        if k.startswith("template") and subsample_templates:
            v = v[templates_select_indices]

        for i, (dim_size, dim) in enumerate(zip(feature_schema_shapes[k], v.shape)):
            is_num_res = dim_size == NUM_RES
            if i == 0 and k.startswith("template"):
                v = v[slice(templates_crop_start, templates_crop_start + num_templates_crop_size)]
            elif is_num_res:
                v = torch.index_select(v, i, crop_idxs)

        protein[k] = v

    protein["seq_length"] = protein["seq_length"].new_tensor(num_res_crop_size)

    return protein
