# Build Instructions

## Environment Setup

### Clone the DeepFold Repository

Begin by cloning the DeepFold repository from GitHub and navigating into the project directory:

```sh
git clone https://github.com/DeepFoldProtein/DeepFold.git
cd DeepFold
```

### Create and Activate the Conda Environment

Set up a new Conda environment with Python 3.11, Ninja build system, and NVIDIA's CUDA toolkit:

```sh
conda create --name deepfold-dev python=3.11 ninja nvidia::cuda-toolkit
conda activate deepfold-dev
```

## CUTLASS Integration (Optional)

CUTLASS (CUDA Templates for Linear Algebra Subroutines and Solvers) is a collection of CUDA C++ template abstractions for high-performance matrix operations. DeepFold utilizes CUTLASS for optimized computations.

### Clone the CUTLASS Repository

Clone the CUTLASS repository into a directory of your choice. Ensure that the version is 3.1.0 or later:

```sh
git clone https://github.com/NVIDIA/cutlass.git
```

### Set the CUTLASS_PATH Environment Variable

Define the CUTLASS_PATH environment variable to point to the CUTLASS directory. Replace `/path/to/cutlass` with the actual path where you cloned CUTLASS:

```sh
export CUTLASS_PATH=/path/to/cutlass
```

Ensure that this environment variable is set in your current session.

## Compilation

### Upgrade Build Tools

Ensure that `pip`, `setuptools`, `wheel`, and `build` are up-to-date:

```sh
python -m pip install --upgrade pip setuptools wheel build
```

### Compile DeepFold

With the `CUTLASS_PATH` environment variable set, proceed to build DeepFold:

```sh
python -m build
```

## Installation

### Install the Compiled Package

Locate the generated `.whl` file in the `dist` directory. The filename will correspond to the version of DeepFold and your Python environment. Install the package using `pip`:

```sh
python -m pip install dist/deepfold-<version>-py311-none-any.whl
```

Replace `<version>` with the actual version number of the package.

**Note:** Ensure that all environment variables are correctly set and that you have the necessary permissions to install packages and compile code on your system.
