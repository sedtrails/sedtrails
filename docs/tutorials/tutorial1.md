# Running a Basic Simulation

This tutorial uses the tracked example configuration at `examples/sedtrails-example.yaml`.

1. From a repository checkout, copy the tracked `examples` folder to the directory where you want to run the model.
```bash
cp -r .../sedtrails/examples <your-runs-folder>
```

2. Download the dataset file named `inlet_sedtrails.nc` from [this link](https://surfdrive.surf.nl/files/index.php/s/VUGKZm7QexAXuD9?path=%2Fdfm).

3. Create a `sample-data` folder next to the copied `examples` folder and place the downloaded dataset there.
```bash
mkdir -p <your-runs-folder>/sample-data
```

4. In `examples/sedtrails-example.yaml`, set `inputs.data` to the downloaded dataset:
```yaml
inputs:
  data: ./sample-data/inlet_sedtrails.nc
```

5. Go to the run directory:
```bash
cd <your-runs-folder>
```

6. Run the model:
```bash
sedtrails run -c ./examples/sedtrails-example.yaml
```

7. SedTRAILS writes the NetCDF output to `examples/results/sedtrails_results.nc` unless you change `outputs.directory` in the configuration.
