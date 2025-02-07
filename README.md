# DeepFold

## Features

- Acceptable(?) replica of AlphaFold model.
- Distributed inference (over multiple GPUs).

## Installation

```sh
git clone git@github.com:DeepFoldProtein/DeepFold.git   # Clone the repository
cd DeepFold                                             # Change directory
poetry install                                          # Build and install the package
```

## Inference

AlphaFold parameter (JAX parameter) is needed to run AlphaFold model of DeepFold framework.

```sh
# Example
INPUT_FEATURES_PKL="out/H1225/features.pkl"
OUTPUT_BASE_DIR="out/H1225"
JAX_PARAMS_DIR="resource/params"

python predict.py \
    --params_dirpath "resources/params" \
    --seed 1 \
    --input_features_filepath out/H1225/features.pkz \
    --output_dirpath out/H1225 \
    --preset params_model_1_multimer_v3 \
    --precision bf16
```

- If you want to enable deterministic mode (for validation) add `--deterministic` flag.
- You can fix feature processing random seed with `-data_random_seed` option.
- You can determine how many GPUs to use with `-nt` flags and `NVIDIA_VISIBLE_DEVICES` environmental variable.

### NCCL

- Multi-GPU inference mode use NCCL (Nvidia Collective Communication Library).
- If the framework stuck on communication, set `NCCL_P2P_DISABLE=1`.
- Turn off ACS(Access Control Services) on BIOS.
- Turn off IOMMU(Input/Output Memory Management Unit) on BIOS to use RDMA/GPUDirect (if your system supports).
- You can disable ACS temporarily by run `scripts/disable_acs.sh` with root permission.

### Environmental variabes

- Set `DEBUG=1` to show debug messages.

## Training

TBA

## Copyright

Copyright 2025 DeepFold Protein Research Team
