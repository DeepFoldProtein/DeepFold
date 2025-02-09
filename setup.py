"""Setup for pip package."""

import os
import sys

import setuptools
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

major = torch.cuda.get_device_properties(0).major
minor = torch.cuda.get_device_properties(0).minor
gpu_arch = f"{major}{minor}"

compute_capabililties = [gpu_arch]
# H100
compute_capabililties.append("90")
# L4, L40, RTX 4090
compute_capabililties.append("89")
# A40, A10, A16, A2, A6000, RTX 3090
compute_capabililties.append("86")
# A100, A30
compute_capabililties.append("80")


generator_flags = []
cc_flag = []
compute_capabililties = list(set(compute_capabililties))


for cap in compute_capabililties:
    cc_flag.append(f"-gencode=arch=compute_{cap},code=sm_{cap}")
    cc_flag.append(f"-gencode=arch=compute_{cap},code=compute_{cap}")


sources = [
    "csrc/evoformer_attn/attention.cpp",
    "csrc/evoformer_attn/attention_back.cu",
    "csrc/evoformer_attn/attention_cu.cu",
]


ext_modules = []

cmdclass = {}

cutlass_path = os.environ.get("CUTLASS_PATH")
if cutlass_path is None:
    print("Please specify the CUTLASS repo directory as environment variable $CUTLASS_PATH", file=sys.stderr)
else:
    includes = [
        f"{cutlass_path}/include",
        f"{cutlass_path}/tools/util/include",
    ]
    includes = [os.path.abspath(p) for p in includes]

    ext_modules.append(
        CUDAExtension(
            name="deepfold_C",
            sources=sources,
            extra_compile_args={
                "cxx": [
                    "-O2",
                    "-std=c++17",
                    *generator_flags,
                ],
                "nvcc": [
                    "-O2",
                    "-std=c++17",
                    "--use_fast_math",
                    f"-DGPU_ARCH={gpu_arch}",
                    "-allow-unsupported-compiler",
                    *generator_flags,
                    *cc_flag,
                ],
            },
            include_dirs=includes,
            optional=False,
        )
    )
    cmdclass["build_ext"] = BuildExtension


setuptools.setup(
    packages=setuptools.find_packages(include=["deepfold", "deepfold.*"]),
    include_package_data=True,
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
