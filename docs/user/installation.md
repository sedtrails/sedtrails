# Installation

:::warning
SedTRAILS is currently in **beta**. Users are welcome to test it, but compatibility may break in future versions. Its use in research requires discretion.
:::

**Requirements:**

SedTRAILS requires Python 3.10 or newer, up to but not including Python 3.14.


## Using pip 
You can install the latest beta version directly from PyPI using pip. Check the [releases page](https://pypi.org/project/sedtrails/#history) for available versions.

```bash
pip install sedtrails==1.0.0-beta.2
```

## From source
To install SedTRAILS from source, follow these steps:

1. Clone the repository:
```bash

git clone https://github.com/sedtrails/sedtrails.git
```

2. Change to the `sedtrails` directory:
```bash
cd sedtrails
```

3. Install the package using `pip` or `conda`:

```bash
# For pip
pip install .
```

```bash
# For conda
# Create and activate the conda environment
conda env create -f environment.yml
conda activate sedtrails
```

## Confirming the installation

Writing the following command in the terminal will show the installed SedTRAILS version:
```bash
sedtrails -v
# E.g.: SedTRAILS 1.0.0-beta.2
```

The following command provides help about the SedTRAILS commands:
```bash
sedtrails -h

```

Check the [Simulations Guide](./simulations.md) to learn how to set up and run a simulation.

