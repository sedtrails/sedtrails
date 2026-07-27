import math
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

__all__ = [
    'DiffusionStrategy',
    'BrownianDiffusionStrategy',
    'DiffusionCalculator',
]


class DiffusionStrategy(ABC):
    """Abstract base class for diffusion strategies."""

    @abstractmethod
    def calculate(
        self,
        dt: float,
        x: np.ndarray,
        y: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        kh: float,
        rng: np.random.Generator | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply diffusion to the given positions and velocities.

        Parameters
        ----------
        dt : float
            Time step for diffusion calculation.
        x : np.ndarray
            Array of x-coordinates.
        y : np.ndarray
            Array of y-coordinates.
        u : np.ndarray
            Array of x-velocity components.
        v : np.ndarray
            Array of y-velocity components.
        kh : float
            Diffusion coefficient.
        rng : np.random.Generator, optional
            Random-number generator for reproducible diffusion.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Updated x and y positions after diffusion.
        """
        pass


class BrownianDiffusionStrategy(DiffusionStrategy):
    """Isotropic Brownian diffusion using a constant kh coefficient."""

    def calculate(
        self,
        dt: float,
        x: np.ndarray,
        y: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        kh: float,
        rng: np.random.Generator | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return calculate.

        Parameters
        ----------
        dt : float
            Integration timestep in seconds.
        x : np.ndarray
            X coordinate value or array.
        y : np.ndarray
            Y coordinate value or array.
        u : np.ndarray
            X velocity component.
        v : np.ndarray
            Y velocity component.
        kh : float
            Horizontal diffusivity value or array.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Array containing the computed values.
        """
        if kh == 0.0 or dt == 0.0:
            return x.copy(), y.copy()

        sigma = math.sqrt(2.0 * kh * dt)
        normal = np.random.standard_normal if rng is None else rng.standard_normal
        dx = sigma * normal(x.shape)
        dy = sigma * normal(y.shape)
        return x + dx, y + dy


class DiffusionCalculator:
    """
    Main class for calculating diffusion effects.
    """

    def __init__(self, strategy: DiffusionStrategy, rng: np.random.Generator | None = None):
        """Initialize with a diffusion strategy.

        Parameters
        ----------
        strategy : DiffusionStrategy
            Diffusion strategy object to use.
        rng : np.random.Generator, optional
            Population-specific random-number generator.
        """
        self._strategy = strategy
        self._rng = rng

    @property
    def strategy(self) -> DiffusionStrategy:
        """
        Get the current diffusion strategy.

        Returns
        -------
        DiffusionStrategy
            The strategy value.
        """
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: DiffusionStrategy):
        """Change the diffusion strategy.

        Parameters
        ----------
        strategy : DiffusionStrategy
            New diffusion strategy object to use.
        """
        self._strategy = strategy

    def calc_diffusion(
        self,
        x: np.ndarray,
        y: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        kh: float,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate diffusion effect on position.

        Parameters
        ----------
        x : np.ndarray
            Current x-coordinate array.
        y : np.ndarray
            Current y-coordinate array.
        u : np.ndarray
            X-velocity array matching ``x`` and ``y``.
        v : np.ndarray
            Y-velocity array matching ``x`` and ``y``.
        kh : float
            Horizontal diffusivity coefficient.
        dt : float
            Current time step in seconds.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Updated ``(x, y)`` coordinate arrays.
        """
        return self._strategy.calculate(dt, x, y, u, v, kh, rng=self._rng)
