# SedTRAILS Plugins

SedTRAILS supports a plugin system that allows users to extend its functionality by 
writing plugins for the *Transport Converter*. 
Plugins can be written for the Format and Physics Converters. 

This document provides an overview of how to create and integrate plugins into SedTRAILS. We use the case of creating a plugin for the Physics Converter as an example.

## Creating a Plugin

We assume you have set up a development environment for SedTRAILS. If you haven't done this yet, please refer to the [Developer Guide](dev-environment.md).
In essence, a plugin for the  Physics Converter must implement a class that inherits from `BasePhysicsPlugin` and implements the `add_physics` method.
The `add_physics` method takes as input a `SedtrailsData` object, which contains data to perform the physics calculations, and add the results (physics conversiont) to the `SedtrailsData` object itself.
To **create a physics plugin for SedTRAILS**, follow these steps:

1. Create a new Python file for your plugin in `src/sedtrails/transport_converter/plugins/physics/`. For example, `myplugin.py`.

2. Define a class for your plugin that inherits from the appropriate base class. For a Physics Converter plugin, inherit from `BasePhysicsPlugin`.

    ```python
    # myplugin.py
    from sedtrails.transport_converter.plugins import BasePhysicsPlugin
    from sedtrails.transport_converter import SedtrailsData # Import SedtrailsData 

    class PhysicsPlugin(BasePhysicsPlugin):
    """
    Plugin for sediment transport physics calculations.
    This plugin implements the physics calculations as described in XXXX.
    """

    def __init__(
        self,
        config
    ):  
        super().__init__()
        self.config = config

    def add_physics(
        self,
        sedtrails_data: SedtrailsData,
        # other arguments
        # other keyword arguments
    ):
        """
        Add physics using XX approach.
        """
        print('Using XX compute transport velocities and add to SedTRAILS data...')

        ## Your to convert transport velocities and add them to the SedtrailsData object
    ```

`config` is a dataclass that contains configuration parameters for your plugin. You can define it according to your needs.
The `add_physics` method is where you implement the logic for your physics calculations. You can access and modify the `SedtrailsData` object to add the results of your calculations to it. More arguments and keyword arguments can be added as needed.

3. Write unit tests for your plugin to ensure it works as expected. Place your tests in the `tests` directory, following the existing structure.

## Registering a Plugin

Once you have created your plugin, you need to register it in the JSON schema in `src/sedtrails/config/population.schema.json` for the simulation configuration file.
To register your plugin: 

1. Add a new definition in the `$defs` section of the schema. The naming convention for the definition should be `<your_plugin_name>_method`. And proceed to define the names data types of the parameters your plugin will required, under `properties`. Such parameters will be passed as attributes of the `config` dataclass in the plugin class. User will only be able to set these parameters in the simulation configuration file. For example, if your plugin is named `my_plugin`, you would add the following to the `$defs` section:

```json
"$defs": {
"my_plugin_method": {
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "parameter1": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "the name of a parameter for my plugin"
                },
                "parameter2": {
                    "type": "number",
                    "default": 0.2,
                    "description": "the name of another parameter"
                }
            },
            "description": "a description of my plugin method"
        }
}
```

2. Add the name of your plugin to section `tracer_methods` and a reference to its parameter definition. This will enable uses to use the plugin in the configuration file. **Important:** *The name of the entry must match the name of the Python file where the plugin is implemented.* Therefore, use distinctive and short names for your plugins files to avoid name clashes.
 Continuing with the `myplugin.py` example, you would add it as follows:

```json

 "tracer_methods": {
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "vanwesten": {
                    "$ref": "#/$defs/van_westen_method"
                },
                "vanwesten_bl": {
                    "$ref": "#/$defs/van_westen_bl_method"
                },
                "soulsby": {
                    "$ref": "#/$defs/soulsby_method"
                },
                "myplugin": {
                    "$ref": "#/$defs/my_plugin_method"
                }
            }
}
```

3. Test your plugin has been correctly registered by adding it to a simulation configuration file and validating the file against the schema. For example, for a configuration file `my-config.yaml`:

```yaml
general:
  input_model: 
    format: fm_netcdf
    reference_date: 1970-01-01  # Default reference date for the input model
    morfac: 1  # Morphological acceleration factor for time decompression
inputs:
  data: ./sample-data/inlet_sedtrails.nc
time:
  start:  2016-09-21 19:20:00
  timestep: 60S
  duration: 1D
  cfl_condition: 0.7  # CFL condition for adaptive timestep (0 = disabled)
particles:
  populations:
    - name: populaton_1
      particle_type: sand
      characteristics:
        grain_size: 0.00025 
        density: 2650.0  
      tracer_methods:
        myplugin:  # the name of the plugin for physics conversion
            parameter1: 'value'  # names and values of the parameters defined in the schema for 'myplugin'
            parameter2: 0.5
      transport_probability: stochastic_transport  # Options: no_probability, stochastic_transport, reduced_velocity
      seeding:
        burial_depth: 
          constant: 0
        release_start: 2016-09-21 19:30:00
        quantity: 1
        strategy: 
          random:
            bbox: "39400,16800 40600,17800"
            seed: 42
            nlocations: 1
outputs:
  directory: ./results
```

```python
# To validate the configuration file with your plugin, you can use the following code snippet:
from sedtrails.simulation import Simulation

sim = Simulation('my-config.yaml') # configuration file using your plugin
sim.validate_config()  # this should pass without errors

```

:::important
Validation of the configuration file will fail if the plugin is not correctly registered in the JSON schema, but not if the plugin itself has errors. Make sure to test your plugin thoroughly.
If you need help, please reach out the [SedTRAILS Team in GitHub](https://github.com/sedtrails/sedtrails/issues).
:::
