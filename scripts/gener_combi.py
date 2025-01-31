#!/usr/bin/env python3

import argparse
import collections
import dataclasses
import itertools
import logging
import os
import re
import string
import sys
from pathlib import Path
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, s):
        for fp in self.files:
            fp.write(s)

    def flush(self):
        for fp in self.files:
            fp.flush()


def get_features_filepath(
    chain_dirpath: Path,
    run_name: str,
) -> Path:
    model_relpath = Path(os.path.split(run_name)[0])
    if not model_relpath.is_absolute():
        model_abspath = chain_dirpath.joinpath(model_relpath)
    else:
        model_abspath = model_relpath.absolute()
    pkz_filepath = model_abspath.joinpath("features.pkz")  # .resolve()
    return pkz_filepath.relative_to(chain_dirpath)


@dataclasses.dataclass(frozen=True)
class ClusterInfo:
    chain_id: str
    name: str
    pkz_path: Path
    cluster_id: int
    score: float

    def __lt__(self, other: "ClusterInfo") -> bool:
        return self.score < other.score


def parse_stoi(stoi_str: str):
    m = re.findall("[A-Z]([1-9][0-9]*)", stoi_str)
    return list(map(int, m))


def parse_cluster(chain_dirpath: Path) -> List[ClusterInfo]:
    chain_id = chain_dirpath.name
    cluster_filepath = chain_dirpath.joinpath("ranking_cluster.txt")
    with open(cluster_filepath, "r") as fp:
        lines = fp.read().strip().splitlines()

    entries = []
    for line in lines:
        if line.startswith("disorder"):
            continue
        _, name, _, plddt, cluster_id = line.strip().split()
        entries.append(
            ClusterInfo(
                chain_id=chain_id,
                name=name,
                pkz_path=get_features_filepath(chain_dirpath, name),
                cluster_id=-1 if cluster_id == "NA" else int(cluster_id),
                score=float(plddt),
            )
        )
    return entries


def deduplicate_and_truncate_clusters(
    target_dirpath: Path,
    chain_id: str,
    cutoff: float = 70.0,
    minimum: int = 5,
) -> List[ClusterInfo]:
    entries = parse_cluster(target_dirpath.joinpath(chain_id))
    entries.sort(reverse=True)
    clusters = collections.defaultdict(list)
    for c in entries:
        clusters[c.cluster_id].append(c)
    for v in clusters.values():
        v.sort(reverse=True)
    best_ones = [v[0] for i, v in enumerate(clusters.values()) if v[0].score >= cutoff or i < minimum]
    # bets_ones  # Sorted
    return best_ones


@dataclasses.dataclass(frozen=True)
class ResultPairs:
    names: List[str]
    score: float = 0.0
    comment: str = ""

    def __lt__(self, other: "ResultPairs") -> bool:
        return self.score < other.score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "target_id",
        help="Target id.",
    )
    parser.add_argument(
        "stoichiom",
        help="Stoichiometry of the complex.",
    )
    parser.add_argument(
        "--output_dirpath",
        default=Path.cwd(),
        help="Output directory path.",
    )
    parser.add_argument(
        "--plddt_cutoff",
        default=70.0,
        type=float,
        dest="cutoff",
        help="plDDT cutoff.",
    )
    parser.add_argument(
        "--num_min",
        default=5,
        type=int,
        help="Minimum number of monomer featuers to combine.",
    )
    args = parser.parse_args()
    return args


def main(args: argparse.Namespace):
    target_id = args.target_id
    cardinality = parse_stoi(args.stoichiom)  # args.stoichiom.split(",")
    xid = range(1, len(cardinality) + 1)

    chain_ids = [f"T{target_id[1:]}s{i}" for i in xid]
    target_dirpath = Path.cwd()
    assert len(cardinality) <= len(string.ascii_uppercase), len(cardinality)
    stoichiom = "".join(f"{c}{n}" for c, n in zip(string.ascii_uppercase, cardinality))
    output_dirpath = target_dirpath.joinpath(f"out/{target_id}")
    output_dirpath.mkdir(parents=True, exist_ok=True)

    # dict is ordered? YES!

    cluster_map = {
        cid: deduplicate_and_truncate_clusters(
            target_dirpath,
            cid,
            cutoff=args.cutoff,
            minimum=args.num_min,
        )
        for cid in chain_ids
    }

    already_seen = set()
    pairs = []
    results = []
    for cx in itertools.product(*cluster_map.values()):
        key = tuple(map(lambda c: str(c.pkz_path), cx))
        if key in already_seen:
            continue
        else:
            pairs.append(cx)
            already_seen.add(key)

        for pair in pairs:
            names = list(map(lambda c: "/".join(c.name.split("/")[:-1]), pair))
            scores = np.fromiter(map(lambda c: c.score, pair), dtype="float")
            clu_ids = list(map(lambda c: c.cluster_id, pair))
            comment = ""
            comment += ", ".join(map(lambda x: f"{x:5.02f}", scores))
            comment += " | "
            comment += ", ".join(map(lambda x: f"{x:2d}", clu_ids))

            result = ResultPairs(
                names=names,
                score=scores[:].mean(),
                comment=comment,
            )
            results.append(result)

    already_seen = set()
    results.sort(reverse=True)

    # with open(output_dirpath.joinpath("note.multimer"), "w") as fp:
    tee = Tee(sys.stdout)
    print(target_id, stoichiom, "#", ",".join(map(str, cardinality)), file=tee)
    print(f"#{0:2d}", *["AF0" for _ in range(len(cardinality))], "#", file=tee)
    for i, r in enumerate(results, start=1):
        key = tuple(r.names)
        if key in already_seen:
            continue
        else:
            print(f"#{i:2d}", *r.names, "#", f"{r.score:5.02f}", "|", r.comment, file=tee)
            already_seen.add(key)


if __name__ == "__main__":
    main(parse_args())
