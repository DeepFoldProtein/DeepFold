"""High‑level prediction orchestrator."""

from __future__ import annotations

import gc
import json
import logging
import os
import random
import signal
import time
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.distributed
from tqdm.auto import tqdm

import deepfold.distributed as df_dist
from deepfold.config import MONOMER_OUTPUT_SHAPES, MULTIMER_OUTPUT_SHAPES, AlphaFoldConfig, FeatureConfig
from deepfold.data.process.pipeline import example_to_features
from deepfold.distributed import model_parallel as mp
from deepfold.modules import inductor as df_inductor
from deepfold.utils.crop_utils import unpad_to_schema_shape_
from deepfold.utils.file_utils import load_pickle
from deepfold.utils.import_utils import import_jax_weights_
from deepfold.utils.log_utils import setup_logging
from deepfold.utils.random import NUMPY_SEED_MODULUS
from deepfold.utils.tensor_utils import tensor_tree_map
from deepfold.utils.torch_utils import disable_tf32, enable_tf32

from .hooks import BaseHooks, DefaultHooks
from .presets import get_preset_opts

logger = logging.getLogger(__name__)


torch.set_grad_enabled(False)


class Predictor:  # noqa: D101
    def __init__(self, args: Any, *, hooks: Optional[BaseHooks] = None) -> None:
        self.args = args
        self.hooks = hooks or DefaultHooks(
            output_dirpath=args.output_dirpath,
            save_recycle=args.save_recycle,
            suffix=f"_{args.suffix}" if args.suffix else "",
            model_name="",
            preset=args.preset,
            seed=args.seed,
            benchmark=args.benchmark,
        )

        # ------------------------------------------------------------------
        # Seeds – identical semantics to v0.1
        # ------------------------------------------------------------------
        if (self.args.seed == -1) and (self.args.mp_size == 0):
            self.seed = random.randint(0, NUMPY_SEED_MODULUS)
        else:
            self.seed = self.args.seed

        # ------------------------------------------------------------------
        # Preset mapping now yields typed config models
        # ------------------------------------------------------------------
        self.model_name, (self._model_kwargs, self._feat_kwargs, self._import_kwargs) = get_preset_opts(self.args.preset)
        self.hooks.model_name = self.model_name  # patch DefaultHooks

        # Override recycling via CLI flag after preset construction
        if self.args.max_recycling_iters >= 0:
            self._feat_kwargs["max_recycling_iters"] = self.args.max_recycling_iters

        self.suffix = f"_{self.args.suffix}" if self.args.suffix else ""

        # ------------------------------------------------------------------
        # Device & distributed init
        # ------------------------------------------------------------------
        self._init_distributed()

        # Lazily initialised state vars
        self._model: Optional[torch.nn.Module] = None
        self._feat_config: Optional[FeatureConfig] = None
        self._model_config: Optional[AlphaFoldConfig] = None

    # =====================================================================
    # Public API
    # =====================================================================
    def run(self) -> None:  # noqa: D401
        """End‑to‑end execution with a tidy progress bar."""
        timings: dict[str, float] = {}

        feats: dict | None = None
        batch: dict[str, torch.Tensor] | None = None
        batch_last: dict | None = None
        outputs: dict | None = None

        def _time(label: str, fn: Callable[[], Any]) -> None:  # noqa: D401
            t0 = time.perf_counter()
            fn()
            timings[label] = time.perf_counter() - t0

        def _logging() -> None:
            self._setup_logging()

        def _load() -> dict:
            nonlocal feats
            feats = self._load_features()
            return feats  # for type checkers

        def _batch() -> tuple[dict[str, torch.Tensor], dict]:
            assert feats is not None, "Features must be loaded before batching"
            nonlocal batch, batch_last
            batch, batch_last = self._create_batch(feats)
            return batch, batch_last

        def _init() -> None:
            self._initialize_model()

        def _infer_stage() -> dict:
            assert None not in (batch, batch_last, feats)
            nonlocal outputs
            outputs = self._infer(batch, batch_last, feats)  # type: ignore[arg-type]
            return outputs

        def _post() -> None:
            assert None not in (outputs, batch_last, feats)
            self._post_process(outputs, batch_last, feats)  # type: ignore[arg-type]

        stages: list[tuple[str, Callable[[], Any]]] = [
            ("Logging", _logging),
            ("Load features", _load),
            ("Create batch", _batch),
            ("Init model", _init),
            ("Inference", _infer_stage),
            ("Post-process", _post),
        ]

        # ------------------------------------------------------------------
        # Master rank: tqdm; workers: direct execution
        # ------------------------------------------------------------------
        if df_dist.is_master_process():
            with tqdm(total=len(stages), desc="Pipeline", dynamic_ncols=True, leave=False) as pbar:
                for label, action in stages:
                    pbar.set_description(label)
                    _time(label, action)
                    pbar.update(1)
        else:
            for _label, action in stages:
                action()

        # ------------------------------------------------------------------
        # Dump timings JSON (only master)
        # ------------------------------------------------------------------
        if df_dist.is_master_process():
            timings_path = self.args.output_dirpath / f"timings_{self.model_name}{self.suffix}.json"
            try:
                timings_path.write_text(json.dumps(timings, indent=4), encoding="utf‑8")
                logger.info("Stage timings written to %s", timings_path)
            except Exception:  # pragma: no cover
                logger.exception("Failed to write timings JSON")

        # Finalize
        if self.args.mp_size > 0:
            torch.distributed.barrier(group=mp.group())  # , device_ids=[df_dist.local_rank()])
            df_dist.destroy()

    # ------------------------------------------------------------------
    # Device / distributed
    # ------------------------------------------------------------------
    def _init_distributed(self) -> None:  # noqa: D401
        if self.args.mp_size > 0:
            df_dist.initialize()
            self.process_name = f"dist_rank{df_dist.rank()}"
            self.device = torch.device(f"cuda:{df_dist.local_rank()}")
            assert len(df_dist.train_ranks()) % self.args.mp_size == 0
            mp.initialize(dap_size=self.args.mp_size)
            if df_dist.is_master_process():
                logger.info("Distributed: WORLD_SIZE=%d MP_SIZE=%d", df_dist.world_size(), mp.size())
        else:
            self.process_name = "single_process"
            self.device = torch.device("cuda:0")
            logger.info("Single‑GPU mode")
        torch.cuda.set_device(self.device)

    # ------------------------------------------------------------------
    # Logging setup & TQDM handler
    # ------------------------------------------------------------------
    def _setup_logging(self) -> None:  # noqa: D401
        self.args.output_dirpath.mkdir(parents=True, exist_ok=self.args.force)
        if df_dist.is_main_process():
            logfile = self.args.output_dirpath / f"predict.{self.model_name}{self.suffix}.log"
            setup_logging(logfile)
            logger.info("Arguments:")
            for k, v in vars(self.args).items():
                logger.info("%s=%s", k, v)

    # ------------------------------------------------------------------
    # Features & batch
    # ------------------------------------------------------------------
    def _load_features(self) -> dict[str, Any]:  # noqa: D401
        feats = load_pickle(self.args.input_features_filepath)
        self.seqlen = int(feats["residue_index"].shape[-1])
        self.hooks.after_features(feats)
        return feats

    def _create_batch(self, feats: dict[str, Any]):  # noqa: D401
        if self.args.precision == "bf16":
            input_dtype = torch.bfloat16
        elif self.args.precision == "tf32":
            input_dtype = torch.float32
        elif self.args.precision == "fp32":
            input_dtype = torch.float32
        else:
            raise ValueError(f"Unknown precision={repr(self.args.precision)}")

        self._model_config = AlphaFoldConfig.from_preset(precision=self.args.precision, **self._model_kwargs)
        self._feat_config = FeatureConfig.from_preset(
            preset="predict",
            subsample_templates=self.args.subsample_templates,
            seed=self.seed,
            num_chunks=self.args.mp_size if self.args.mp_size > 0 else 1,
            **self._feat_kwargs,
        )
        if df_dist.is_master_process():
            logger.info("Config resolved – dumping flattened dicts ...")
            from deepfold.utils.iter_utils import flatten_dict

            for k, v in flatten_dict(self._model_config.to_dict()).items():
                logger.info("%s=%s", k, v)
            for k, v in flatten_dict(self._feat_config.to_dict()).items():
                logger.info("%s=%s", k, v)

        t0 = time.perf_counter()
        batch = example_to_features(feats, self._feat_config)
        if df_dist.is_master_process():
            logger.info("Feature processing: %.2f s", time.perf_counter() - t0)

        batch_last = {k: np.array(v[..., -1].numpy()) for k, v in batch.items()}
        batch_tensor: dict[str, torch.Tensor] = {}
        for k, v in batch.items():
            assert torch.is_tensor(v), "example_to_features should return torch tensors"
            target_dtype = input_dtype if torch.is_floating_point(v) else v.dtype

            tensor = v.unsqueeze(0).to(device=self.device, dtype=target_dtype)
            batch_tensor[k] = tensor

        self.hooks.after_batch(batch_tensor)
        return batch_tensor, batch_last

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    def _initialize_model(self):  # noqa: D401
        df_inductor.disable()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state(device=self.device)

        np.random.seed(self.seed % NUMPY_SEED_MODULUS)
        torch.manual_seed(self.seed)

        from deepfold.modules.alphafold import AlphaFold

        model = AlphaFold(config=self._model_config)
        model.eval()
        if self.args.precision == "bf16":
            enable_tf32()
            for name, module in model.named_children():
                if name in ("structure_module", "auxiliary_heads"):
                    continue
                module.to(dtype=torch.bfloat16)
        elif self.args.precision == "tf32":
            enable_tf32()
        elif self.args.precision == "fp32":
            disable_tf32()
        else:
            raise ValueError(f"Unknown precision={repr(self.args.precision)}")
        model.to(device=self.device)

        torch.cuda.set_rng_state(cuda_state, device=self.device)
        torch.set_rng_state(torch_state)
        np.random.set_state(numpy_state)

        npz_path = self.args.params_dirpath / f"{self.args.preset}.npz"
        import_jax_weights_(model=model, npz_path=npz_path, **self._import_kwargs)
        model.eval()
        self.hooks.after_model_init(model)
        self._model = model
        return model

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def _infer(self, batch: dict[str, torch.Tensor], batch_last: dict, feats: dict):  # noqa: D401
        if df_dist.is_master_process():
            logger.info("seqlen=%d --> padded=%d", self.seqlen, batch["seq_mask"].shape[-2])
            logger.info("Start inference ...")

        # --------------------------------------------------------------
        # Inner tqdm for recycling – only on master
        # --------------------------------------------------------------
        recycle_bar = None  # type: Optional[tqdm]
        if df_dist.is_master_process():
            total_recycles: int | None = getattr(self._model_config, "num_recycle", None)
            if total_recycles is None:
                total_recycles = getattr(self._feat_config, "max_recycling_iters", None)
            if total_recycles is not None:
                total_recycles += 1  # include initial pass (recycle 0)
            recycle_bar = tqdm(
                total=total_recycles,
                desc="Recycle",
                position=1,
                dynamic_ncols=True,
                leave=False,
            )

        def recycle_hook(i: int, f: dict, o: dict):  # noqa: D401
            """Proxy to user hook + bar update."""
            self.hooks.on_recycle(i, f, o)
            if recycle_bar is not None:
                recycle_bar.update(1)

        # --------------------------------------------------------------
        # Run forward(s)
        # --------------------------------------------------------------
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        out = self._model(batch, recycle_hook=recycle_hook, save_all=self.args.save_all)
        if recycle_bar is not None:
            recycle_bar.close()

        # --------------------------------------------------------------
        # Book keeping as before
        # --------------------------------------------------------------
        if self.args.mp_size > 0:
            gc.collect()
            torch.cuda.empty_cache()
            torch.distributed.barrier(group=mp.group(), device_ids=[df_dist.local_rank()])
        elapsed = time.perf_counter() - t0
        if df_dist.is_master_process():
            logger.info("Inference finished in %.2f s", elapsed)
            logger.info("CUDA peak memory: %.2f MB", torch.cuda.max_memory_allocated() / 1024 / 1024)
        if df_dist.is_master_process():
            return tensor_tree_map(lambda x: np.array(x.cpu().squeeze(0).numpy()), out)
        return {}

    # ------------------------------------------------------------------
    # Post‑processing
    # ------------------------------------------------------------------
    def _post_process(self, outputs: dict[str, Any], batch_last: dict, feats: dict):  # noqa: D401
        if not df_dist.is_master_process():
            return
        self.hooks.after_prediction(
            outputs,
            processed_features=batch_last,
            seqlen=self.seqlen,
            model_config=self._model_config,
            feat_config=self._feat_config,
            MONOMER_OUTPUT_SHAPES=MONOMER_OUTPUT_SHAPES,
            MULTIMER_OUTPUT_SHAPES=MULTIMER_OUTPUT_SHAPES,
            unpad_to_schema_shape_=unpad_to_schema_shape_,
        )
