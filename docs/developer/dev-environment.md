
# Development Setup

Developers and contributors can set up a local development environment to work on the SedTRAILS codebase. Follow the steps below to get started.

1. Make sure Python 3.10 or newer, up to but not including Python 3.14, and a code editor (e.g., Visual Studio Code) are installed.

1. Fork the SedTRAILS repository on GitHub to your own account. If it is your first time using contritubing to SedTRAILS, please read the [contributing guidelines](contribution.md). 

1. Clone the SedTRAILS repository from github in your desired directory:
```bash
git clone git@github.com:<your-user-name>/sedtrails.git
```

1. Change the directory:
```bash
cd sedtrails
```

1. Change to the development branch:
```bash
git swiltch dev
```

1. Create and activate a virtual environment to manage dependencies using the tool of your choice. For example, using `venv` or `conda`. 

```bash
# If using venv
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`

# If using conda
conda create --name sedtrails-dev python=3.13
conda activate sedtrails-dev

```

1. Install the dependencies: ([more details](https://github.com/sedtrails/sedtrails/blob/dev/CONTRIBUTING.md))
```bash
pip install -e .[dev]
```

1. Confirming the installation  by typing on the terminal:
```bash
sedtrails -v

# The installed version should be displayed
```

For help:
```bash
sedtrails -h
```

## Simulation Test

To fully test your development environment, you can run a sample simulation using the tracked example configuration.

1. Download the dataset file named `inlet_sedtrails.nc` from [this link](https://surfdrive.surf.nl/files/index.php/s/VUGKZm7QexAXuD9?path=%2Fdfm).

2. From the repository root, create a local sample-data directory and place the downloaded dataset there:
```bash
mkdir -p sample-data
```

3. In `examples/sedtrails-example.yaml`, set `inputs.data` to the downloaded dataset:

```yaml
inputs:
  data: ./sample-data/inlet_sedtrails.nc
```

4. Run the model:
```bash
sedtrails run -c examples/sedtrails-example.yaml
```
