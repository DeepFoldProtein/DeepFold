import argparse
import dataclasses
import json
import logging
import string
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_filepath",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "-d",
        "--target_dirpath",
        default=Path.cwd(),
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
    setup_logging("multimer.log")
    return parser.parse_args()


@dataclasses.dataclass(frozen=True)
class Unit:
    feature_filepath: Path
    num_sym: int = 1


@dataclasses.dataclass(frozen=True)
class Recipe:
    name: str
    units: List[Unit] = dataclasses.field(default_factory=Unit)
    version: int = 2


def parse_input_json(input_json: str) -> List[Recipe]:
    o = json.loads(input_json)
    records = []
    for r in o["recipes"]:
        units = []
        for u in r["units"]:
            units.append(
                Unit(
                    feature_filepath=u["path"],
                    num_sym=int(u["num"]),
                )
            )
        rec = Recipe(
            name=r["name"],
            units=units,
        )
        records.append(rec)
    return records


def cook_recipe_v2(
    recipe: Recipe,
    output_dirpath: Path,
    target_dirpath: Path = Path.cwd(),
    force: bool = True,
) -> Dict[str, str]:
    assert recipe.version == 2

    name = recipe.name
    output_dirpath = output_dirpath / name
    output_dirpath.mkdir(parents=True, exist_ok=force)

    units = recipe.units
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
    input_filepath: Path = args.input_filepath
    recipes = parse_input_json(input_filepath.read_text())

    for k, v in vars(args).items():
        logger.info(f"{k}={v}")

    for recipe in recipes:
        dic = cook_recipe_v2(
            recipe=recipe,
            output_dirpath=args.output_dirpath,
            target_dirpath=args.target_dirpath,
        )
        logger.info(dic)


if __name__ == "__main__":
    main(parse_args())
