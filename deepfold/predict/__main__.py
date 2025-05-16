"""Entry‑point wrapper for ``python -m deepfold.predict``."""

from __future__ import annotations

import sys

from .cli import parse_args
from .predictor import Predictor


def _main(argv: list[str] | None = None) -> None:  # noqa: D401
    args = parse_args(argv)
    predictor = Predictor(args)
    try:
        predictor.run()
    except KeyboardInterrupt:  # pragma: no cover
        import logging

        logging.getLogger(__name__).info("Interrupted – exiting with status 1 ...")
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    _main()
