
# Development Environment

## Installation for developers

This is the step-by-step process to install SedTRAILS.

1. Make sure Python (3.13 or newer) and a code editor (e.g., Visual Studio) is installed.

2. Clone the SedTRAILS repository from github in your desired directory:
```bash
cd <your desired directory>
git clone https://github.com/sedtrails/sedtrails.git
```

3. Change the directory:
```bash
cd sedtrails
```

4. Install the dependencies: ([more details](https://github.com/sedtrails/sedtrails/blob/dev/CONTRIBUTING.md))
```bash
pip install -e .[dev]
```

## Confirming the installation

Type the following command in the terminal:
```bash
sedtrails -v
```

For help:
```bash
sedtrails -h

```
## Running an example model

1. Download the dataset file named "inlet_sedtrails.nc" from [this link](https://surfdrive.surf.nl/files/index.php/s/VUGKZm7QexAXuD9?path=%2Fdfm).

2. Change the directory:
```bash
cd examples
```

3. In the file config.example.yaml, update the directory for "input_data" to read the data-set: 
  input_data: ./sample-data/inlet_sedtrails.nc


4. Run the model:
```bash
sedtrails run -c .\config.example.yaml
```