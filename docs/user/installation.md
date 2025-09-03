# Installation

::: warning
Update this section
:::

 SedTRAILS is currently in an alpha stage. It is not recommended for production use.

This is the step-by-step guide on how to install SedTRAILS.

1. Make sure Python (3.13 or newer) and a code editor (e.g., Visual Studio) is installed.

2. Clone the SedTRAILS repository from github in your desired directory:
```bash
cd <your desired directory>
git clone https://github.com/sedtrails/sedtrails.git
```

3. Go to the directory in which the SedTRAILS repository is cloned into:
```bash
cd sedtrails
```

4. Install the dependencies: ([See this webpage](https://github.com/sedtrails/sedtrails/blob/dev/CONTRIBUTING.md) for more details about installing as a developer)

```bash
pip install .
```

## Confirming the installation

Writing the following command in the terminal will show the installed SedTRAILS version:
```bash
sedtrails -v
```

The following command provides help about SedTRAILS installation:
```bash
sedtrails -h

```
## Running an example model

1. copy/paste the exmaples folder (i.e., /sedtrails/examples) in a directory where you want to run the model.
```bash
cp -r ./examples <your runs folder>
```

2. Download the dataset file named "inlet_sedtrails.nc" from [this link](https://surfdrive.surf.nl/files/index.php/s/VUGKZm7QexAXuD9?path=%2Fdfm).

2. Go to the directory in which you want to run the model:
```bash
cd .../<your runs folder>
```

3. In the file config.example.yaml, update the directory for "input_data" to read the data-set you doanloaded in the first step: 
  input_data: ./sample-data/inlet_sedtrails.nc


4. Run the model:
```bash
sedtrails run -c .\config.example.yaml
```

5. The results plots should pop up and should be saved in the ./results directory.