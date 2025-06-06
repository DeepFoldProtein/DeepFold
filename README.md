# DeepFold

## Features

(TBA)

## Installation

```bash
git clone https://github.com/DeepFoldProtein/DeepFold.git
cd DeepFold
pip install .
```

```bash
pip install git+https://github.com/DeepFoldProtein/DeepFold@main
```

## Monomer featurization

The `deepfold-monomer` command wraps the `build_input_features` function to generate all necessary input features for a single‐chain DeepFold run. It reads a query FASTA, optional MSA alignments, and optional template search results, then writes a pickled dictionary containing sequence, MSA, and template features. Below is a description of the available flags, required inputs, and an example invocation.

You will also need:

- A working Kalign binary (or another alignment‐to‐PDB MSA tool) in your `PATH`, unless you specify a custom path with `--kalign-bin`.
- A local copy of all gzipped mmCIF files (for PDB template retrieval), or set `--pdb-mmcif-dir` to wherever you store them.
- (Optional) A PDB obsolete file if you want to filter out deprecated template structures.

### Usage

```bash
deepfold-monomer \
  --fasta     /path/to/query.fasta \
  --alignments /path/to/msa1.a3m /path/to/msa2.sto \
  --template  /path/to/template_search.hhr \
  --output    /path/to/output_features.pkz \
  [--pdb-mmcif-dir   /path/to/mmcif_dir] \
  [--pdb-obsolete    /path/to/pdb_obsolete_file] \
  [--max-template-date YYYY-MM-DD] \
  [--max-template-hits 20] \
  [--kalign-bin $(which kalign)$] \
  [--seed 42]
```

#### Required Arguments

- `-f, --fasta <Path>`
  Path to a FASTA file containing exactly one protein sequence.
  Example: `--fasta query.fasta`

- `-o, --output <Path>`
  Path where the output pickled feature file (`*.pkz`) will be written. The parent directory will be created if it does not exist.
  Example: `--output features/query_features.pkz`

#### Optional Arguments

- `-a, --alignments <Path> [<Path> …]`
  One or more MSA search result files. Supported extensions:

  - `.a3m` (raw A3M)
  - `.sto` (Stockholm-formatted multiple‐sequence alignment)
    These will be parsed and converted into MSA features.
    Example: `--alignments uniref.a3m bfd.sto`

- `-t, --template <Path>`
  A file containing template‐hit results. Supported extensions:

  - `.hhr` (HHsearch output)
  - `.sto` (HMMsearch Stockholm output)

- `--pdb-mmcif-dir <Path>`
  Directory containing gzipped mmCIF files indexed by PDB ID. This is required if your template hits need to be fetched from PDB files.
  Example: `--pdb-mmcif-dir /data/mmcif_files/`

- `--pdb-obsolete <Path>`
  Path to an “obsolete” PDB mapping file (commonly downloaded from the RCSB). Used to filter out deprecated templates.
  Example: `--pdb-obsolete obsolete.dat`

- `--max-template-date <YYYY-MM-DD>` (default: today’s date)
  Only include templates released on or before this date. Use `YYYY-MM-DD` format.
  Example: `--max-template-date 2025-05-01`

- `--max-template-hits <int>` (default: 20)
  Maximum number of template hits to retain.
  Example: `--max-template-hits 10`

- `--kalign-bin <string>` (default: `kalign`)
  Full path to the Kalign executable (used to realign template hits). If `kalign` is already on your `PATH`, the default is sufficient.
  Example: `--kalign-bin /usr/local/bin/kalign`

- `--seed <int>`
  Random seed for shuffling template hits (if multiple comparable hits exist).
  Example: `--seed 1234`

### Output

The output file specified by `--output` is a gzipped pickle (`*.pkz`) containing a Python dictionary with the following keys:

- **Sequence features**

  - `residue_index`: Integer index of each residue (with optional `--offset`).
  - `aatype`, `sequence_features_*`: One‐hot encoding and ancillary features derived from the primary sequence.

- **MSA features**

  - `msa`, `deletion_matrix`, `num_alignments`, etc.
  - (If `--parse-descr` was set, alignment scores and identifiers are included.)

- **Template features**

  - `template_domain_names`: Array of byte‐encoded strings for each template domain (or empty if no templates).
  - `template_sequence`, `template_aatype`: Query‐aligned sequences and one‐hot encoding for each template.
  - `template_all_atom_positions`, `template_all_atom_mask`: 3D coordinates and masks for all atoms in each template, padded to the full query length.
  - `template_sum_probs`: Scalar “confidence” score for each template hit.

### Example

Assume you have:

- `query.fasta` (single‐sequence FASTA)
- Two MSA files: `uniref90.a3m` and `bfd_uniclust.sto`
- A template search result from HHsearch: `query.hhr`
- A local mmCIF directory: `/data/pdb/mmcif/`
- An obsolete PDB list: `/data/pdb/obsolete.dat`

Run:

```bash
deepfold-monomer \
  --fasta        query.fasta \
  --alignments   uniref90.a3m bfd_uniclust.sto \
  --template     query.hhr \
  --pdb-mmcif-dir /data/pdb/mmcif/ \
  --pdb-obsolete  /data/pdb/obsolete.dat \
  --output       target/features.pkz \
  --max-template-date 2025-05-15 \
  --max-template-hits  10 \
  --template-mode hhr \
  --seed          42 \
  --offset        0 \
```

This will:

1. Parse `query.fasta` (must have exactly one sequence).
2. Parse the two MSA files (converting the `.sto` to `.a3m` if needed) and build MSA features, including alignment scores.
3. Read `query.hhr`, extract up to 10 best template hits no later than 2025-05-15, realign them with Kalign, and featurize them.
4. Write a consolidated pickled feature dictionary to `target/features.pkz`.

Once complete, you can feed `target/features.pkz` directly into the DeepFold monomer model.

## Inference

AlphaFold/DeepFold parameters (JAX parameter) are needed to run DeepFold framework.

The prediction runner CLI processes input features (pickled feature dictionaries) and runs a DeepFold model to generate structural predictions. Below is a description of the available command-line arguments and an example invocation.

### Usage

```bash
deepfold-predict \
  --input-features   /path/to/input_features.pkz \
  --output-dir       /path/to/output_directory/ \
  --params-dir       /path/to/parameter_archives/ \
  --preset           <preset_key> \
  [--seed SEED] \
  [--mp-size MP_SIZE] \
  [--precision {fp32|bf16|tf32}] \
  [--max-recycling-iters N] \
  [--suffix SUFFIX] \
  [--force] \
  [--save-recycle] \
  [--save-all] \
  [--exclude-template-torsion-angles] \
  [--subsample-templates] \
  [--benchmark]
```

> **Note:** The exact CLI name (`deepfold-predict` above) may vary depending on how the package’s entry point is defined. Substitute with the appropriate command if different.

### Required Arguments

- `-i, --input-features <Path>`
  Path to the pickled feature file (e.g., `.pkz`) produced by the feature‐builder step (sequence/MSA/template features).

  ```bash
  --input-features /data/features/query_features.pkz
  ```

- `-o, --output-dir <Path>`
  Directory where prediction outputs will be written. If it does not exist, it will be created.

  ```bash
  --output-dir /results/query_prediction/
  ```

- `-p, --params-dir <Path>`
  Directory containing one or more `.npz` parameter archives for the DeepFold model. The runner will load model weights from this directory.

  ```bash
  --params-dir /models/deepfold_params/
  ```

- `--preset <string>`
  Model preset key. Must be one of the keys defined in `deepfold.presets.VALID_PRESETS`.

  ```bash
  --preset evolution_v1
  ```

### Optional Fine-Tuning & Runtime Parameters

- `--seed <int>` (default: -1)
  Global random seed. Use `-1` to pick a random seed at runtime; otherwise, set an integer for reproducibility.

  ```bash
  --seed 42
  ```

- `--mp-size <int>` (default: 0)
  Tensor-parallel group size. Valid values are `0` (disable tensor parallelism), `1`, `2`, `4`, or `8`.

  ```bash
  --mp-size 2
  ```

- `--precision {fp32, bf16, tf32}` (default: `fp32`)
  Floating-point precision for inference.

  - `fp32`: standard 32-bit floats
  - `bf16`: bfloat16
  - `tf32`: TensorFloat-32 (NVIDIA Ampere and later)

  ```bash
  --precision bf16
  ```

- `--max-recycling-iters <int>` (default: -1)
  Override the number of recycling iterations used by the model. If set to `-1`, the runner uses the default value specified by the chosen preset.

  ```bash
  --max-recycling-iters 3
  ```

- `--suffix <string>` (default: `""`)
  Suffix appended to all output filenames (e.g., model checkpoints, PDB files).

  ```bash
  --suffix _run1
  ```

- `--force`
  If set, overwrite any existing contents in the output directory. Otherwise, the runner will error if the directory is non-empty.

  ```bash
  --force
  ```

### Boolean Flags

- `--save-recycle`
  If enabled, write a separate PDB file after each recycling iteration (can be useful for debugging or analyzing intermediate structures).

- `--save-all`
  Save all internal MSA and pair representations into the final output pickle (results in a larger file, but provides complete model state for later analysis).

- `--exclude-template-torsion-angles`
  Do not include template torsion angles in the template‐featurization stage. May be useful if you want to ignore template angular information.

- `--subsample-templates`
  When multiple template hits are available, randomly subsample instead of using the top‐ranked templates. Useful for ensembling or testing robustness.

- `--benchmark`
  Skip writing any large output files (e.g., full pickle) and run only the minimal steps needed to measure runtime performance. Use this flag if you want to measure inference speed without saving full results.

### Example

Below is a complete example that runs DeepFold prediction with a specific preset, using bfloat16 precision, saving all intermediates, and writing per-recycle PDBs:

```bash
deepfold-predict \
  --input-features   target/features.pkz \
  --output-dir       target/ \
  --params-dir       parmas/ \
  --preset           deepfold_model_1 \
  --seed             1234 \
  --precision        tf32 \
  --max-recycling-iters 3 \
  --force
```

1. **Loads** `target/features.pkz` (pickled feature dict).
2. **Selects** the `deepfold_model_1` preset (weight files are pulled from `params/deepfold_model1_1.npz`).
3. **Runs** inference using tf32 precision.
4. **Recycles** up to 3+1 times.
5. **Overwrites** any existing files in `target/` (due to `--force`).

After completion, the `output-dir` will contain:

- `unrelaxed_model_1.pdb` (unrelaxed final structure)
- `results_model_1.pkl` (pickled model outputs, if not in benchmark mode)

You can then visualize or further analyze these results using your preferred structural biology tools or scripts.

### NCCL

- Multi-GPU inference mode use NCCL (Nvidia Collective Communication Library).
- If the framework stuck on communication, set `NCCL_P2P_DISABLE=1`.
- Turn off ACS(Access Control Services) on BIOS.
- Turn off IOMMU(Input/Output Memory Management Unit) on BIOS to use RDMA/GPUDirect (if your system supports).
- You can disable ACS temporarily by run `scripts/disable_acs.sh` with root permission.

## Training

TBA

## Citation

```bibtex
@article{Lee2023,
    title = {DeepFold: enhancing protein structure prediction through optimized loss functions,  improved template features,  and re-optimized energy function},
    volume = {39},
    ISSN = {1367-4811},
    url = {http://dx.doi.org/10.1093/bioinformatics/btad712},
    DOI = {10.1093/bioinformatics/btad712},
    number = {12},
    journal = {Bioinformatics},
    publisher = {Oxford University Press (OUP)},
    author = {Lee,  Jae-Won and Won,  Jong-Hyun and Jeon,  Seonggwang and Choo,  Yujin and Yeon,  Yubin and Oh,  Jin-Seon and Kim,  Minsoo and Kim,  SeonHwa and Joung,  InSuk and Jang,  Cheongjae and Lee,  Sung Jong and Kim,  Tae Hyun and Jin,  Kyong Hwan and Song,  Giltae and Kim,  Eun-Sol and Yoo,  Jejoong and Paek,  Eunok and Noh,  Yung-Kyun and Joo,  Keehyoung},
    editor = {Elofsson,  Arne},
    year = {2023},
    month = nov
}
```

## Copyright

Copyright 2025 DeepFold Protein Research Team
