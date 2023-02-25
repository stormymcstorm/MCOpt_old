# MCOpt

## Requirements

Beyond the dependencies listed in `setup.cfg`, this project also requires 
[topologytoolkit](https://topology-tool-kit.github.io/) be installed inorder to 
generate morse complexes.

## Running the demos
1. **Setup a virtual environment**
```bash
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
```
2. **Install the package**
```bash
pip install -e .
```
3. **Generate or retrieve the data**

The data can be generated by running
```bash
python python generate_data.py
```
Otherwise, get the zipped data from [here](https://github.com/stormymcstorm/MCOpt/releases/download/v0.2.0/gen_data.zip) and unzip it at the root directory. You should have a `gen_data` directory.

4. **Run a demo notebook**
