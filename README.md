# CS450-Project

## Setup

### Clone the Repository

To clone the project with all submodules:

```bash
git clone --recurse-submodules https://github.com/andreslavescu/CS450-Project.git
cd CS450-Project
```

If you already cloned the repository without submodules, initialize them with the following:

```bash
git submodule update --init --recursive
```

### Install Modal

Install the Modal CLI:

```bash
pip install modal
```

Authenticate with Modal:

```bash
modal setup
```

### Docker Image

The project uses GPU-specific Dockerfiles:
- `Dockerfile.h100` - H100 GPU env
- `Dockerfile.b200` - B200 GPU env

Both images:
- Use CUDA 13.0 with cuDNN
- Install PyTorch with CUDA 13.0 support
- Copy and install the Megakernels library
- Build the llama3 1b demo that target the respective GPU (B200 or H100)

Modal will automatically build and cache this Docker image on first run. The image will only rebuild in the modal instance if you modify the Dockerfile.

### Running on Modal

The project supports two implementations:

1. **Hazy baseline** (`--hazy-megakernel`) - runs the HazyResearch reference implementation for megakernel
2. **Waterloo implementation** (`--waterloo-megakernel`) - runs our baseline for the project

```bash
modal run run_modal.py --hazy-megakernel --gpu h100 # run application targetting h100 to reproduce hazy baseline 
modal run run_modal.py --waterloo-megakernel --gpu h100 # run application targetting h100 to reproduce waterloo baseline
modal run run_modal.py --hazy-megakernel --gpu b200 # run application targetting b200 to reproduce hazy baseline 
modal run run_modal.py --waterloo-megakernel --gpu b200 # run application targetting b200 to reproduce waterloo baseline
```

## Relevant Links

[Pull Request Template](.github/PULL_REQUEST_TEMPLATE.md)

[Issue Templates](.github/ISSUE_TEMPLATE/)

[Contributing Rules](.github/CONTRIBUTING.md)
