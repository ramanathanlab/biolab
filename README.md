# biolab
Protein/DNA Language Model benchmarks

## Installation

To install the package, clone and run the following command:
```bash
pip install -U pip setuptools wheel
pip install -e .
```

To install the CaLM benchmark:
```bash
pip install git+https://github.com/oxpig/CaLM.git
```

## Usage
To run a benchmark, pass the appropriate YAML config file to this command:
```bash
python -m biolab.evaluate --config [PATH]
```

## Contributing

For development, it is recommended to use a virtual environment. The following
commands will create a virtual environment, install the package in editable
mode, and install the pre-commit hooks.
```bash
python -m venv venv
source venv/bin/activate
pip install -U pip setuptools wheel
pip install -e '.[dev]'
pre-commit install
```
To test the code, run the following command:
```bash
pre-commit run --all-files
tox -e py310
```
