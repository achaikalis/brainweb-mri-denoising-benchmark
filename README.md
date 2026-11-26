# Brainweb MRI Denoising Benchmark

MRI Denoising Benchmark using the Brainweb Dataset.

## Setup

Install Python version 3.12 and the required dependencies of this repository:

```bash
uv python install 3.12
uv sync --python 3.12 --all-groups
```

Enable the virtual environment and activate Jupyter Lab:

```bash
source .venv/bin/activate
uv run python -m ipykernel install --user --name brainweb-mri-denoising-benchmark --display-name "Python (brainweb-mri-denoising-benchmark)"
uv run jupyter lab
```

Check linting with Ruff:

```bash
uv run ruff check
```

## MINC Toolkit

The minc-convert dependency is offered by conda and thus lives on a separate virtual environment. Setup by doing:

```bash
conda create -n minc-convert -c conda-forge -c minc-forge minc-toolkit-v2
conda activate minc-convert
which mnc2nii
mnc2nii -help
```

@inproceedings{cocosco1997brainweb,
  author    = {Cocosco, Chris A. and Kollokian, Vasken and Kwan, Remi K.-S. and Evans, Alan C.},
  title     = {{BrainWeb}: Online Interface to a {3D} {MRI} Simulated Brain Database},
  booktitle = {NeuroImage, Proceedings of 3rd International Conference on Functional Mapping of the Human Brain},
  year      = {1997},
  volume    = {5},
  number    = {4, part 2/4},
  pages     = {S425},
  address   = {Copenhagen},
  month     = {May}
}
