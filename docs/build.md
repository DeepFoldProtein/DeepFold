# Build

## Environment

```sh
git clone https://github.com/DeepFoldProtein/DeepFold.git
cd DeepFold
conda create --name deepfold-dev python=3.11 ninja nvidia::cuda-toolkit
conda activate deepfold-dev
```

## CUTLASS

Clone CUTLASS version >= 3.1.0 to `CUTLASS_PATH`.

```sh
git clone https://github.com/NVIDIA/cutlass
```

## Compile

```sh
python -m pip install --upgrade pip setuptools wheel build
CUTLASS_PATH="${CUTLASS_PATH}" python -m build
```
