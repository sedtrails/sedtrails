import os
import importlib.util
import inspect
import pytest
from sedtrails.transport_converter.plugins.physics.plugin import BasePhysicsPlugin

# TODO: Is there a better way to get the plugin directory?
PLUGIN_DIR = os.path.dirname(__file__).replace(
    'tests/transport_converter/plugins', 'src/sedtrails/transport_converter/plugins/physics'
    )

def get_plugin_classes():
    """
    Dynamically discovers and imports all plugin classes in the physics plugin directory,
    excluding the base plugin and dunder files.

    Returns:
        list: List of plugin classes found in the directory.
    """    
    plugin_classes = []
    for fname in os.listdir(PLUGIN_DIR):
        if fname.endswith('.py') and fname != 'plugin.py' and not fname.startswith('__'):
            module_path = os.path.join(PLUGIN_DIR, fname)
            module_name = f"sedtrails.transport_converter.plugins.physics.{fname[:-3]}"
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            for _, obj in inspect.getmembers(module, inspect.isclass):
                # Only consider classes defined in this module (not imported ones)
                if obj.__module__ == module_name:
                    plugin_classes.append(obj)
    return plugin_classes

@pytest.mark.integration
@pytest.mark.parametrize("plugin_class", get_plugin_classes())
def test_plugin_inherits_base(plugin_class):
    """
    Integration test to ensure each plugin inherits from BasePhysicsPlugin.
    """
    module_parts = plugin_class.__module__.split('.')
    plugin_file = module_parts[-1] + ".py" if module_parts else "unknown"
    assert issubclass(plugin_class, BasePhysicsPlugin), (
        f"{plugin_file} does not inherit from BasePhysicsPlugin"
    )

@pytest.mark.integration
@pytest.mark.parametrize("plugin_class", get_plugin_classes())
def test_plugin_has_add_physics(plugin_class):
    """
    Integration test to ensure each plugin has an add_physics method.
    """
    # Try to extract from module name
    module_parts = plugin_class.__module__.split('.')
    plugin_file = module_parts[-1] + ".py" if module_parts else "unknown"
    assert hasattr(plugin_class, "add_physics"), (
        f"{plugin_file} does not have an add_physics method"
    )

@pytest.mark.integration
@pytest.mark.parametrize("plugin_class", get_plugin_classes())
def test_plugin_add_physics_callable(plugin_class):
    """
    Integration test to ensure each plugin's add_physics method is callable.
    """
    # Try to extract from module name
    module_parts = plugin_class.__module__.split('.')
    plugin_file = module_parts[-1] + ".py" if module_parts else "unknown"
    method = getattr(plugin_class, "add_physics", None)
    assert callable(method), (
        f"add_physics method in {plugin_file} is not callable"
    )

@pytest.mark.integration
@pytest.mark.parametrize("plugin_class", get_plugin_classes())
def test_plugin_add_physics_accepts_sedtrails_data(plugin_class):
    """
    Integration test to ensure each plugin's add_physics method accepts sedtrails_data as parameter.
    """
    # Try to extract from module name
    module_parts = plugin_class.__module__.split('.')
    plugin_file = module_parts[-1] + ".py" if module_parts else "unknown"
    
    method = getattr(plugin_class, "add_physics", None)
    assert method is not None, f"{plugin_file} does not have an add_physics method"
    
    sig = inspect.signature(method)
    params = list(sig.parameters.values())
    
    # The first parameter should be 'self', the second should be 'sedtrails_data'
    assert len(params) >= 2, (
        f"add_physics method in {plugin_file} should accept at least 'self' and 'sedtrails_data' as arguments"
    )
    assert params[1].name == "sedtrails_data", (
        f"add_physics method in {plugin_file} should take `sedtrails_data` as second parameter, got '{params[1].name}'"
    )