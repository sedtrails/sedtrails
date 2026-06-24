import math
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

__all__ = [
    "DiffusionStrategy",
    "GradientDiffusionStrategy",
    "RandomDiffusionStrategy",
    "BrownianDiffusionStrategy",
    "DiffusionCalculator",
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
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply diffusion to the given positions and velocities.

           Parameters
           ----------
            dt: Time step for diffusion calculation
            x: Array of x-coordinates.
            y: Array of y-coordinates.
            u: Array of x-velocity components.
            v: Array of y-velocity components.
            kh: Diffusion coefficient.

        Returns
        -------
            Tuple of updated x and y positions after diffusion (xdif, ydif).
        """
        pass


class GradientDiffusionStrategy(DiffusionStrategy):
    """
    Diffusion based on spatial gradients of the velocity field.
    """

    def calculate(
        self,
        dt: float,
        x: np.ndarray,
        y: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        kh: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Example dummy diffusion calculation using gradients of the velocity field.

        For unstructured grids, gradients are approximated using nearest-neighbor differences.

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
        n = len(x)
        dudx = np.zeros(n)
        dudy = np.zeros(n)
        dvdx = np.zeros(n)
        dvdy = np.zeros(n)

        for i in range(n):
            distances = np.sqrt((x - x[i]) ** 2 + (y - y[i]) ** 2)
            if np.all(distances == 0):  # Only one point
                continue
            distances[i] = np.inf
            nearest_idx = np.argmin(distances)
            dx = x[nearest_idx] - x[i]
            dy = y[nearest_idx] - y[i]
            if dx != 0 or dy != 0:
                scale = 1.0 / (dx**2 + dy**2 + 1e-10)
                dudx[i] = (u[nearest_idx] - u[i]) * dx * scale
                dudy[i] = (u[nearest_idx] - u[i]) * dy * scale
                dvdx[i] = (v[nearest_idx] - v[i]) * dx * scale
                dvdy[i] = (v[nearest_idx] - v[i]) * dy * scale

        # Diffusion terms: kh * Laplacian
        xdif = kh * (dudx + dvdy)
        ydif = kh * (dudy + dvdx)
        return xdif, ydif


class RandomDiffusionStrategy(DiffusionStrategy):
    """
    Random walk diffusion model.
    """

    def calculate(
        self,
        dt: float,
        x: np.ndarray,
        y: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        kh: float,
    ) -> Tuple[float, float]:
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
        Tuple[float, float]
            Floating-point result of the calculation.
        """
        vel_mag = np.sqrt(u**2 + v**2)
        rndnr_mag = np.random.randn(*vel_mag.shape)
        mag_diff = np.abs(rndnr_mag * kh) * vel_mag
        rndnr_angle = np.random.rand(*vel_mag.shape)
        angle_diff = rndnr_angle * 2 * math.pi
        dx_diff = mag_diff * np.cos(angle_diff) * dt
        dy_diff = mag_diff * np.sin(angle_diff) * dt
        xdif = x + dx_diff
        ydif = y + dy_diff

        return xdif, ydif


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
        sigma = math.sqrt(2.0 * kh * dt)
        dx = np.random.normal(0.0, sigma, size=x.shape)
        dy = np.random.normal(0.0, sigma, size=y.shape)
        return x + dx, y + dy


class DiffusionCalculator:
    """
    Main class for calculating diffusion effects.
    """

    def __init__(self, strategy: DiffusionStrategy):
        """Initialize with a diffusion strategy.

        Parameters
        ----------
            strategy: DiffusionStrategy object to use
        """
        self._strategy = strategy

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
            strategy: New DiffusionStrategy object to use
        """
        self._strategy = strategy

    def calc_diffusion(
        self,
        x: float,
        y: float,
        u: np.ndarray,
        v: np.ndarray,
        kh: float,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate diffusion effect on position.

        Parameters
        ----------
            x: Current x position
            y: Current y position
            u: Velocity field x-component (2D array)
            v: Velocity field y-component (2D array)
            kh: Diffusion coefficient
            dt: Current time step

        Returns
        -------
            Tuple of (x_diffusion, y_diffusion) representing position changes
        """
        return self._strategy.calculate(dt, x, y, u, v, kh)
