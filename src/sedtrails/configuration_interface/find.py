"""
A convenience function to find a value in the configuration data using a dot-separated key.
"""

from typing import Any, Dict, Union


def find_value(data: Dict, keys: str, default=None) -> Any:
    """
    Retrieves a value from a dictionary using a dot-separated notation for nested dictionaries.

    Parameters
    ----------
    data : Dict
        The dictionary to search in.
    key : str
        The dot-separated key to retrieve from the configuration.
    default : any, optional
        The default value to return if the key is not found.

    Returns
    -------
    any
        The value associated with the key or the default value. None if the key is not found.

    Example
    --------
    >>> config = {
            'population': {
                'seeding': {
                    'strategy': {'point': {'locations': ['1.0,2.0', '3.0,4.0']}},
                    'quantity': 10,
                }
            }
        }
    >>> point = find('population.seeding.strategy.point')
    >>> print(point))  # Outputs `{'locations': ['1.0,2.0', '3.0,4.0']}`

    """

    current: Union[Dict, Any] = data
    for key in keys.split('.'):
        if not isinstance(current, dict):
            return default
        current = current.get(key, default)
        if not isinstance(current, dict):
            return current
    return current
