[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![TUDelft DCC](https://img.shields.io/badge/tu_delft-DCC-black?style=flat&label=TU%20Delft&labelColor=%23000000%20&color=%2300A6D6)](https://dcc.tudelft.nl)
[![Deploy Sphinx Documentation](https://github.com/sedtrails/sedtrails/actions/workflows/publish.yml/badge.svg)](https://github.com/sedtrails/sedtrails/actions/workflows/publish.yml)
[![Ruff](https://github.com/sedtrails/sedtrails/actions/workflows/ruff.yml/badge.svg?branch=dev)](https://github.com/sedtrails/sedtrails/actions/workflows/ruff.yml)
[![Pytest](https://github.com/sedtrails/sedtrails/actions/workflows/pytest.yml/badge.svg?branch=dev&event=push)](https://github.com/sedtrails/sedtrails/actions/workflows/pytest.yml)

# SedTRAILS
**Sed**iment **TRA**nsport v**I**sualization and **L**agrangian **S**imulator.

SedTRAILS is an open-source Python package for modeling sediment transport that uses Lagrangian particle tracking to simulate sediment pathways in coastal and estuarine environments. The current version is a **beta release**, bugs and issues are expected. Please report any problems you encounter on the [GitHub Issues page](https://github.com/sedtrails/sedtrails/issues)

## Features
- Lagrangian particle tracking for sediment transport simulation.
- Dashboard for interactive visualization of simulation results in real-time.
- Support for D-Flow FM, XBeach, and SFINCS hydrodynamic model outputs.
- Support for various physics convertion methods.
- Terminal user interface (CLI) for easy setup and execution of simulations.
- Modular design for easy integration and extension.
- Comprehensive documentation and examples.


## Installation

**Requirements:**
- SedTRAILS requires Python 3.10 or newer, up to but not including Python 3.14.


### Using pip 
You can install the latest beta version directly from PyPI using pip. Check the [releases page](https://pypi.org/project/sedtrails/#history) for available versions.

```bash
pip install sedtrails==1.0.0-beta.2
```

### From Source
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

4. Verify the installation by running:
```bash
sedtrails --version

# You should see the installed version number.
# E.g.: SedTRAILS 1.0.0-beta.2
```

## Contributing Guidelines
We welcome contributions from the community! Read the [contributing guidelines](CONTRIBUTING.md) to know how can you take part in developing SedTRAILS. 

## License
SedTRAILS is licensed under the [GPL 3.0 License](LICENSE).
&copy; (2025) SedTRAILS Team, Delft, The Netherlands. 

## Citation

Pearson, S. G., Reyns, J., Garcia Alvarez, M. G., Urhan, A., van Westen, B., Pannozzo, N., & Shafiei, H. SedTRAILS (Version 1.0.0-beta.2) [Computer software]

### Author Contributions
| [Role](https://credit.niso.org/contributor-roles-defined/) | Author                                                                                                |
| ---------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| Conceptualization                                          | Garcia Alvarez, M., Pannozzo N., Pearson, S. G., Reyns J., Shafiei, H., Urhan, A., & van Westen, B.   |
| Project management                                         | Garcia Alvarez, M., & Pearson, S. G.                                                                  |
| Investigation                                              | Pannozzo N., Pearson, S. G., Reyns J., Shafiei, H., & van Westen, B.                                  |
| Software                                                   | Garcia Alvarez, M., Pannozzo N., Pearson, S. G., Reyns J.,  Shafiei, H., Urhan, A., &  van Westen, B. |
| Visualization                                              | Pearson, S. G., & van Westen, B.                                                                      |
| Supervision                                                | Garcia Alvarez, M., Pearson, S., & Reyns, J.                                                          |
| Funding acquisition                                        | Pearson, S. G., & Reyns J.                                                                            |


## Acknowlegdements

> - The design and development of *SedTRAILS Software* was supported by the [Digital Competence Centre](https://dcc.tudelft.nl/), Delft University of Technology.
> - We want to thank **Monica Aguilera Chaves** from [Deltares](https://www.deltares.nl/en/) for partcipating in the initial design discussions of SedTRAILS.

