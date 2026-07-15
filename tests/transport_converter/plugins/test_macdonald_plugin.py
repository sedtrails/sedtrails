"""Regression tests for MacDonald plugin diagnostics."""

import warnings

import numpy as np
import pytest

from sedtrails.transport_converter.plugins.physics.macdonald import PhysicsPlugin


def test_lookup_invalid_input_emits_filterable_warning_once(monkeypatch, capsys):
    """Invalid lookup values should warn once without printing to stdout."""
    monkeypatch.setattr(PhysicsPlugin, "_macdonald_warned_oob", False)

    with pytest.warns(RuntimeWarning, match="Some inputs were invalid"):
        result = PhysicsPlugin.calculate_macdonald_susp_load_height(0.0, 1.0)

    assert np.isnan(result)
    assert capsys.readouterr().out == ""

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        PhysicsPlugin.calculate_macdonald_susp_load_height(0.0, 1.0)

    assert caught == []