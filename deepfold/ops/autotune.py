import builtins
import json
import logging
import os
import sys
import time
from importlib.metadata import version
from multiprocessing import Lock
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import platformdirs
import torch
import triton

logger = logging.getLogger(__name__)
FILE_LOCK = Lock()
FORCE_TUNE = os.getenv("TRIFAST_FORCE_TUNE", "0").lower() in ("1", "true", "yes", "on")

device_capability = torch.cuda.get_device_capability()
device_capability_str = f"{device_capability[0]}-{device_capability[1]}"
device_name = torch.cuda.get_device_name().replace(" ", "-")


def get_config_dir() -> Path:
    """
    Returns the user configuration directory for DeepFold, creating it if it does not exist.

    Returns:
        Path: The path to the configuration directory.
    """
    path = Path(platformdirs.user_config_dir(appname="deepfold", version=version("deepfold")))
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return path


CONFIG_DIR = get_config_dir()


def config_to_dict(config: triton.Config) -> Dict[str, Any]:
    """
    Convert a Triton Config object to a dictionary representation.

    Args:
        config (triton.Config): The Triton configuration to serialize.

    Returns:
        dict: A dictionary containing the config's kwargs, num_warps, and num_stages.
    """
    return {
        "kwargs": config.kwargs,
        "num_warps": config.num_warps,
        "num_stages": config.num_stages,
    }


def dict_to_config(config_dict: Dict[str, Any]) -> triton.Config:
    """
    Convert a dictionary to a Triton Config object.

    Args:
        config_dict (dict): A dictionary with keys 'kwargs', 'num_warps', and 'num_stages'.

    Returns:
        triton.Config: The reconstructed Triton configuration.
    """
    return triton.Config(
        kwargs=config_dict["kwargs"],
        num_warps=config_dict["num_warps"],
        num_stages=config_dict["num_stages"],
    )


# Default fusion configurations
_FWD_CONFIGS = [
    triton.Config(kwargs={"BLOCK_J": 16, "BLOCK_K": 32}, num_warps=1, num_stages=2),
    triton.Config(kwargs={"BLOCK_J": 32, "BLOCK_K": 16}, num_warps=1, num_stages=2),
    triton.Config(kwargs={"BLOCK_J": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    triton.Config(kwargs={"BLOCK_J": 64, "BLOCK_K": 16}, num_warps=2, num_stages=3),
    triton.Config(kwargs={"BLOCK_J": 32, "BLOCK_K": 32}, num_warps=1, num_stages=1),
    triton.Config(kwargs={"BLOCK_J": 128, "BLOCK_K": 16}, num_warps=2, num_stages=2),
    triton.Config(kwargs={"BLOCK_J": 128, "BLOCK_K": 32}, num_warps=2, num_stages=2),
    triton.Config(kwargs={"BLOCK_J": 32, "BLOCK_K": 64}, num_warps=2, num_stages=2),
    triton.Config(kwargs={"BLOCK_J": 64, "BLOCK_K": 32}, num_warps=2, num_stages=3),
    triton.Config(kwargs={"BLOCK_J": 32, "BLOCK_K": 16}, num_warps=1, num_stages=4),
]
if FORCE_TUNE:
    _FWD_CONFIGS.extend(
        [
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 32}, num_warps=1, num_stages=5),
            triton.Config({"BLOCK_J": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_J": 128, "BLOCK_K": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 64}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_J": 64, "BLOCK_K": 16}, num_warps=2, num_stages=3),
            triton.Config({"BLOCK_J": 128, "BLOCK_K": 16}, num_warps=2, num_stages=2),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 16}, num_warps=2, num_stages=4),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 16}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 64}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK_J": 128, "BLOCK_K": 16}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 32}, num_warps=8, num_stages=1),
            triton.Config({"BLOCK_J": 64, "BLOCK_K": 32}, num_warps=8, num_stages=1),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 16}, num_warps=2, num_stages=5),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 16}, num_warps=8, num_stages=1),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 64}, num_warps=8, num_stages=1),
            triton.Config({"BLOCK_J": 64, "BLOCK_K": 64}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK_J": 128, "BLOCK_K": 32}, num_warps=8, num_stages=1),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 32}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 128}, num_warps=4, num_stages=2),
        ]
    )

_BWD_Q_CONFIGS = [
    triton.Config({"BLOCK_J": 16, "BLOCK_K": 32}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_J": 32, "BLOCK_K": 16}, num_warps=1, num_stages=2),
    triton.Config({"BLOCK_J": 32, "BLOCK_K": 64}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_J": 32, "BLOCK_K": 32}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK_J": 64, "BLOCK_K": 16}, num_warps=2, num_stages=3),
    triton.Config({"BLOCK_J": 16, "BLOCK_K": 16}, num_warps=1, num_stages=2),
    triton.Config({"BLOCK_J": 32, "BLOCK_K": 16}, num_warps=1, num_stages=3),
    triton.Config({"BLOCK_J": 16, "BLOCK_K": 64}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK_J": 32, "BLOCK_K": 64}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_J": 64, "BLOCK_K": 32}, num_warps=2, num_stages=2),
]
if FORCE_TUNE:
    _BWD_Q_CONFIGS.extend(
        [
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 128}, num_warps=2, num_stages=2),
            triton.Config({"BLOCK_J": 64, "BLOCK_K": 16}, num_warps=2, num_stages=3),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 64}, num_warps=1, num_stages=1),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 16}, num_warps=1, num_stages=3),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 32}, num_warps=2, num_stages=3),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 128}, num_warps=2, num_stages=3),
            triton.Config({"BLOCK_J": 128, "BLOCK_K": 16}, num_warps=2, num_stages=2),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 16}, num_warps=1, num_stages=4),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 64}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK_J": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_J": 128, "BLOCK_K": 32}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 32}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 32}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_J": 64, "BLOCK_K": 16}, num_warps=1, num_stages=2),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 64}, num_warps=2, num_stages=3),
            triton.Config({"BLOCK_J": 128, "BLOCK_K": 16}, num_warps=1, num_stages=2),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 32}, num_warps=1, num_stages=3),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 128}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK_J": 64, "BLOCK_K": 32}, num_warps=1, num_stages=2),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 128}, num_warps=4, num_stages=2),
        ]
    )


_BWD_KV_CONFIGS = [
    triton.Config({"BLOCK_J": 32, "BLOCK_K": 32}, num_warps=1, num_stages=2),
    triton.Config({"BLOCK_J": 32, "BLOCK_K": 16}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_J": 64, "BLOCK_K": 16}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_J": 32, "BLOCK_K": 32}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_J": 32, "BLOCK_K": 32}, num_warps=1, num_stages=3),
    triton.Config({"BLOCK_J": 64, "BLOCK_K": 32}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_J": 16, "BLOCK_K": 16}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_J": 32, "BLOCK_K": 16}, num_warps=1, num_stages=2),
    triton.Config({"BLOCK_J": 32, "BLOCK_K": 32}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK_J": 16, "BLOCK_K": 32}, num_warps=2, num_stages=1),
]

if FORCE_TUNE:
    _BWD_KV_CONFIGS.extend(
        [
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 128}, num_warps=2, num_stages=2),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 32}, num_warps=1, num_stages=4),
            triton.Config({"BLOCK_J": 64, "BLOCK_K": 16}, num_warps=2, num_stages=3),
            triton.Config({"BLOCK_J": 128, "BLOCK_K": 16}, num_warps=2, num_stages=3),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 64}, num_warps=2, num_stages=3),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 64}, num_warps=1, num_stages=1),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 16}, num_warps=1, num_stages=3),
            triton.Config({"BLOCK_J": 64, "BLOCK_K": 32}, num_warps=2, num_stages=4),
            triton.Config({"BLOCK_J": 64, "BLOCK_K": 64}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK_J": 128, "BLOCK_K": 32}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 128}, num_warps=2, num_stages=1),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 32}, num_warps=1, num_stages=3),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 64}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK_J": 64, "BLOCK_K": 16}, num_warps=1, num_stages=2),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 64}, num_warps=2, num_stages=4),
            triton.Config({"BLOCK_J": 128, "BLOCK_K": 16}, num_warps=1, num_stages=2),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 128}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK_J": 64, "BLOCK_K": 32}, num_warps=1, num_stages=2),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 16}, num_warps=4, num_stages=2),
        ]
    )

_BWD_B_CONFIGS = [
    triton.Config({"BLOCK_J": 16, "BLOCK_K": 16}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_J": 16, "BLOCK_K": 16}, num_warps=2, num_stages=3),
    triton.Config({"BLOCK_J": 16, "BLOCK_K": 32}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_J": 32, "BLOCK_K": 32}, num_warps=1, num_stages=2),
    triton.Config({"BLOCK_J": 32, "BLOCK_K": 64}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_J": 16, "BLOCK_K": 64}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_J": 16, "BLOCK_K": 16}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_J": 32, "BLOCK_K": 32}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_J": 16, "BLOCK_K": 16}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_J": 32, "BLOCK_K": 32}, num_warps=2, num_stages=3),
]

if FORCE_TUNE:
    _BWD_B_CONFIGS.extend(
        [
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 64}, num_warps=2, num_stages=4),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 64}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 32}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 32}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 16}, num_warps=2, num_stages=6),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 16}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 128}, num_warps=2, num_stages=4),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 64}, num_warps=2, num_stages=4),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 32}, num_warps=2, num_stages=5),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 16}, num_warps=2, num_stages=6),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 64}, num_warps=1, num_stages=2),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 32}, num_warps=2, num_stages=5),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 16}, num_warps=1, num_stages=6),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 64}, num_warps=1, num_stages=3),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 32}, num_warps=1, num_stages=4),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 64}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 16}, num_warps=1, num_stages=4),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 128}, num_warps=1, num_stages=3),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 32}, num_warps=1, num_stages=6),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 16}, num_warps=4, num_stages=4),
        ]
    )


class Autotuner(triton.runtime.Autotuner):
    """
    A Triton Autotuner that caches tuning results to disk for persistence.

    Overrides the base Triton Autotuner to store and load tuning cache.
    """

    def __init__(
        self,
        fn: Any,
        arg_names: List[str],
        configs: List[triton.Config],
        key: List[str],
        reset_to_zero: Optional[List[str]],
        restore_value: Optional[List[str]],
        pre_hook: Optional[Callable] = None,
        post_hook: Optional[Callable] = None,
        prune_configs_by: Optional[Dict[str, Callable]] = None,
        warmup: Optional[int] = None,
        rep: Optional[int] = None,
        use_cuda_graph: bool = False,
    ) -> None:
        """
        Initialize the Autotuner and load existing cache if available.

        Args:
            fn: The Triton JIT function to autotune.
            arg_names: Names of function arguments used for keying cache.
            configs: List of Triton Config objects to evaluate.
            key: Argument names whose values trigger config reevaluation.
            reset_to_zero: Args to reset before evaluating configs.
            restore_value: Args to restore after evaluating configs.
            pre_hook: Optional function called before kernel execution.
            post_hook: Optional function called after kernel execution.
            prune_configs_by: Functions for pruning config search space.
            warmup: Warmup time (ms) for benchmarking.
            rep: Repetition count (ms) for benchmarking.
            use_cuda_graph: Whether to use CUDA graph captures.
        """
        super().__init__(
            fn,
            arg_names,
            configs,
            key,
            reset_to_zero,
            restore_value,
            pre_hook,
            post_hook,
            prune_configs_by,
            warmup,
            rep,
            use_cuda_graph,
        )
        self.cache_file: Optional[Path] = None
        if CONFIG_DIR:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            filename = f"{fn.__name__}_{device_name}_{device_capability_str}.json"
            self.cache_file = CONFIG_DIR / filename
            if self.cache_file.exists():
                try:
                    with FILE_LOCK, open(self.cache_file, "rb") as f:
                        raw = json.load(f)
                        self.cache = {k: dict_to_config(v) for k, v in raw.items()}
                except Exception as e:
                    logger.warning("Could not load autotune cache from %s: %s", self.cache_file, e)
                    self.cache = {}

    def _write_cache_to_disk(self) -> None:
        """
        Persist the current tuning cache to disk atomically.
        """
        if not self.cache_file:
            return
        temp_path = Path(f"{self.cache_file}.{os.getpid()}.tmp")
        try:
            with FILE_LOCK:
                with open(temp_path, "w") as f:
                    json.dump({k: config_to_dict(v) for k, v in self.cache.items()}, f, indent=4)
                os.replace(temp_path, self.cache_file)
        except Exception as e:
            logger.warning("Failed to write autotune cache: %s", e)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """
        Run the autotuned function, benchmarking and caching as needed.

        Args:
            *args: Positional arguments to the Triton kernel.
            **kwargs: Keyword arguments to the Triton kernel.

        Returns:
            Any: The return value of the Triton kernel run.
        """
        self.nargs = dict(zip(self.arg_names, args))
        used_cache = True
        if len(self.configs) > 1:
            # Build a cache key based on provided key args and dtypes
            key_parts: List[str] = []
            for arg in self.keys:
                val = self.nargs.get(arg) or kwargs.get(arg)
                if val is not None:
                    key_parts.append(str(val))
            for val in list(self.nargs.values()) + list(kwargs.values()):
                if hasattr(val, "dtype"):
                    key_parts.append(str(val.dtype))
            cache_key = "_".join(key_parts)
            if cache_key not in self.cache or FORCE_TUNE:
                used_cache = False
                pruned = self.prune_configs(kwargs)
                start = time.time()
                timings = {cfg: self._bench(*args, config=cfg, **kwargs) for cfg in pruned}
                self.bench_time = time.time() - start
                best = builtins.min(timings, key=timings.get)
                self.cache[cache_key] = best
                self.pre_hook({**self.nargs, **kwargs, **best.all_kwargs()}, reset_only=True)
                self.configs_timings = timings
                self._write_cache_to_disk()
            config = self.cache[cache_key]
        else:
            config = self.configs[0]
        self.best_config = config
        if os.getenv("TRITON_PRINT_AUTOTUNING") == "1" and not used_cache:
            print(
                f"Triton autotuning for {self.base_fn.__name__} finished in {self.bench_time:.2f}s; " f"best config: {self.best_config}.",
                file=sys.stderr,
            )
        if config.pre_hook:
            config.pre_hook({**self.nargs, **kwargs, **config.all_kwargs()})
        result = self.fn.run(*args, **kwargs, **config.all_kwargs())
        self.nargs = None
        return result


def autotune(
    configs: List[triton.Config],
    key: List[str],
    prune_configs_by: Optional[Dict[str, Callable]] = None,
    reset_to_zero: Optional[List[str]] = None,
    restore_value: Optional[List[str]] = None,
    pre_hook: Optional[Callable] = None,
    post_hook: Optional[Callable] = None,
    warmup: Optional[int] = None,
    rep: Optional[int] = None,
    use_cuda_graph: bool = False,
    do_bench: Optional[Callable] = None,
) -> Callable:
    """
    Decorator to apply autotuning to a Triton JIT function.

    Args:
        configs: List of Triton Config objects to evaluate.
        key: Argument names triggering config re-benchmark on change.
        prune_configs_by: Rules to prune config search space.
        reset_to_zero: Args to reset before config benchmarking.
        restore_value: Args to restore after config benchmarking.
        pre_hook: Hook called before kernel execution.
        post_hook: Hook called after kernel execution.
        warmup: Warmup ms for benchmarking (deprecated).
        rep: Repetition ms for benchmarking (deprecated).
        use_cuda_graph: Whether to use CUDA graph captures.
        do_bench: Custom benchmarking function.

    Returns:
        Callable: A decorator that replaces the function with an Autotuner instance.
    """

    def decorator(fn: Any) -> Autotuner:
        return Autotuner(
            fn,
            fn.arg_names,
            configs,
            key,
            reset_to_zero,
            restore_value,
            pre_hook=pre_hook,
            post_hook=post_hook,
            prune_configs_by=prune_configs_by,
            warmup=warmup,
            rep=rep,
            use_cuda_graph=use_cuda_graph,
        )

    return decorator
