import argparse
import dataclasses
import json
import logging
import string
import sys
from pathlib import Path
from typing import Dict, List

from matplotlib import pyplot as plt

from deepfold.data.multimer.input_features import ComplexInfo, process_multimer_features
from deepfold.eval.plot import plot_msa
from deepfold.utils.file_utils import dump_pickle, load_pickle
from deepfold.utils.log_utils import setup_logging

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


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--show-example",
        action="store_true",
    )
    parser.add_argument(
        "-i",
        "--input_filepath",
        type=Path,
    )
    parser.add_argument(
        "-d",
        "--target_dirpath",
        type=Path,
    )
    parser.add_argument(
        "-o",
        "--output_dirpath",
        default=Path.cwd(),
        type=Path,
    )
    parser.add_argument(
        "-l",
        "--log_filepath",
        default=Path.cwd().joinpath("list.multimer"),
        type=Path,
    )
    parser.add_argument(
        "--pairing",
        default="none",
        choices=["none"],
    )
    return parser


@dataclasses.dataclass(frozen=True)
class Entity:
    feature_filepath: Path
    num_sym: int = 1


@dataclasses.dataclass(frozen=True)
class Structures:
    name: str
    entities: List[Entity] = dataclasses.field(default_factory=Entity)
    version: int = 2


def parse_input_json(input_json: str) -> List[Structures]:
    o = json.loads(input_json)
    records = []
    for r in o["structures"]:
        units = []
        for u in r["entities"]:
            units.append(
                Entity(
                    feature_filepath=u["path"],
                    num_sym=int(u["num_sym"]),
                )
            )
        rec = Structures(
            name=r["name"],
            entities=units,
        )
        records.append(rec)
    return records


def cook_recipe_v2(
    structures: Structures,
    output_dirpath: Path,
    target_dirpath: Path = Path.cwd(),
    force: bool = True,
) -> Dict[str, str]:
    assert structures.version == 2

    name = structures.name
    output_dirpath = output_dirpath / name
    output_dirpath.mkdir(parents=True, exist_ok=force)

    units = structures.entities
    stoich = ""

    descriptions = []
    num_units = []
    all_monomer_features = {}
    for alph, unit in zip(string.ascii_uppercase, units):
        stoich += f"{alph:s}{unit.num_sym:d}"
        descriptions.append(alph)
        num_units.append(unit.num_sym)
        monomer_feature = load_pickle(target_dirpath / unit.feature_filepath)
        all_monomer_features[alph] = monomer_feature

    example = process_multimer_features(
        complex=ComplexInfo(
            descriptions=descriptions,
            num_units=num_units,
        ),
        all_monomer_features=all_monomer_features,
    )

    output_filepath = output_dirpath / "features.pkz"
    dump_pickle(example, output_filepath)

    fig = plot_msa(example)
    depth_filepath = output_dirpath / "msa_depth.png"
    fig.savefig(depth_filepath)
    plt.close(fig)

    return {"name": name, "stoich": stoich, "output_dirpath": str(output_dirpath)}


def main(args: argparse.Namespace):
    if args.show_example:
        example = {
            "structures": [
                {
                    "name": "multimer",
                    "entities": [
                        {
                            "path": "features/chainA.pkz",
                            "num_sym": 2,
                        },
                        {
                            "path": "features/chainB.pkz",
                            "num_sym": 3,
                        },
                    ],
                }
            ]
        }
        print(json.dumps(example, indent=4))
        return

    input_filepath: Path = args.input_filepath
    recipes = parse_input_json(input_filepath.read_text())

    for k, v in vars(args).items():
        logger.info(f"{k}={v}")

    if args.target_dirpath is None:
        target_dirpath = input_filepath.parent
    else:
        target_dirpath = args.target_dirpath

    for recipe in recipes:
        dic = cook_recipe_v2(
            structures=recipe,
            output_dirpath=args.output_dirpath,
            target_dirpath=target_dirpath,
        )
        logger.info(dic)


if __name__ == "__main__":
    parser = get_parser()
    setup_logging("multimer.log")
    args = parser.parse_args()

    if not args.show_example and args.input_filepath is None:
        parser.error("--input_filepath is required unless --show_example is specified.")
        sys.exit(2)

    main(args)
