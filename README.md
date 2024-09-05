# biolab
Protein/DNA Language Model benchmarks

## Contributing

For development, it is recommended to use a virtual environment. The following
commands will create a virtual environment, install the package in editable
mode, and install the pre-commit hooks.
```bash
python -m venv venv
source venv/bin/activate
pip install -U pip setuptools wheel
pip install -e '.[dev,docs]'
pre-commit install
```
To test the code, run the following command:
```bash
pre-commit run --all-files
tox -e py310
```