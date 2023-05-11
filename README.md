# MCOpt

## Requirements

Beyond the dependencies listed in `setup.cfg`, this project also requires 
[topologytoolkit](https://topology-tool-kit.github.io/) be installed inorder to 
generate morse complexes.

## Usage
1. **Setup a virtual environment**
```bash
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
```

2. **Install the package**
```bash
pip install -e .
```

## Experiments
It is recommend that you use the include [devcontainer](https://code.visualstudio.com/docs/devcontainers/containers)
to run the experiments. In order to run all experiments, run the following
```bash
cd experiments; python pipeline.py
```