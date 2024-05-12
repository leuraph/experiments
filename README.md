# experiments

This repository serves as a collection of numerical experiments performed.
This allows for reproducability of results in a clean way.

Each experiment is contained in s ingle directory.

## Python experiments
The Python version used in an experiment is indicated in the corresponding `.python-version` file.
For the Python version management, we suggest [`pyenv`](https://github.com/pyenv/pyenv).

1) Change to the directory containing the experiment of interest.
2) Install the Python version specified in the `.python-version` file.
    > Using `pyenv`, this is as simple as `pyenv install <version>`
3) Using the Python version specified, create a virtual environment, e.g. `python -m venv .venv --prompt $(basename "$(pwd)")`
4) Install all the requirements needed for the experiment with `pip install -r requirements.txt`.
    > If you have troubles installing some of the requirements because of lacking permissions to any of my repositories,
    > please do not hesitate to reach out.