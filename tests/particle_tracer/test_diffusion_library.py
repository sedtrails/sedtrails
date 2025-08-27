import numpy as np

from sedtrails.particle_tracer.diffusion_library import (
    DiffusionCalculator,
    GradientDiffusion,
    RandomDiffusion,
)


class TestDiffusion:
    """
    Test suite for diffusion strategies and the diffusion calculator.

    This class contains tests for:
    - The deterministic GradientDiffusion strategy.
    - The stochastic RandomDiffusion strategy.
    - The DiffusionCalculator class and its strategy switching.

    Methods
    -------
    test_gradient_diffusion_calculation():
        Test the GradientDiffusion strategy returns output arrays with
        the correct shape and type.
    test_random_diffusion_calculation():
        Test the RandomDiffusion strategy returns output arrays with
        the correct shape and determinism when seeding the RNG.
    test_diffusion_calculator_strategy():
        Test that DiffusionCalculator correctly uses the provided
        diffusion strategy and allows switching strategies.
    """

    def setup_method(self):
        """
        Set up common test data for diffusion strategy tests.

        This method creates simple arrays for x, y, u, and v that are used
        in the tests for both GradientDiffusion and RandomDiffusion.
        """
        # Create a simple grid and corresponding velocity arrays.
        self.x = np.array([0.0, 1.0, 2.0])
        self.y = np.array([0.0, 1.0, 0.0])
        self.u = np.array([1.0, 2.0, 3.0])
        self.v = np.array([1.0, 1.0, 1.0])
        self.dt = 0.1
        self.nu = 0.5

    def test_gradient_diffusion_calculation(self):
        """
        Test the GradientDiffusion strategy.

        Verifies that the calculate method returns two NumPy arrays of the same shape as the input.
        """
        strategy = GradientDiffusion()
        xdif, ydif = strategy.calculate(
            self.dt, self.x, self.y, self.u, self.v, self.nu
        )

        # Check that the returned values are NumPy arrays.
        assert isinstance(xdif, np.ndarray), "xdif should be a numpy array."
        assert isinstance(ydif, np.ndarray), "ydif should be a numpy array."

        # Check that the output shapes match the input shape.
        assert xdif.shape == self.x.shape, "xdif shape mismatch."
        assert ydif.shape == self.y.shape, "ydif shape mismatch."

    def test_random_diffusion_calculation(self):
        """
        Test the RandomDiffusion strategy.

        Verifies that the calculate method returns two NumPy arrays of the same shape as the input.
        Checks for deterministic behavior when seeding the random number generator.
        """
        strategy = RandomDiffusion()

        # Set a fixed random seed and calculate diffusion.
        np.random.seed(42)
        xdif1, ydif1 = strategy.calculate(
            self.dt, self.x, self.y, self.u, self.v, self.nu
        )

        # Reset the seed and calculate again to ensure reproducibility.
        np.random.seed(42)
        xdif2, ydif2 = strategy.calculate(
            self.dt, self.x, self.y, self.u, self.v, self.nu
        )

        # Check that the returned values are NumPy arrays.
        assert isinstance(xdif1, np.ndarray), "xdif should be a numpy array."
        assert isinstance(ydif1, np.ndarray), "ydif should be a numpy array."

        # Check that the output shapes match the input shape.
        assert xdif1.shape == self.x.shape, "xdif shape mismatch."
        assert ydif1.shape == self.y.shape, "ydif shape mismatch."

        # Check that the outputs are the same with the same seed.
        assert np.allclose(xdif1, xdif2), (
            "Random diffusion x output not reproducible with fixed seed."
        )
        assert np.allclose(ydif1, ydif2), (
            "Random diffusion y output not reproducible with fixed seed."
        )

    def test_diffusion_calculator_strategy(self):
        """
        Test the DiffusionCalculator class with different strategies.

        Verifies that DiffusionCalculator correctly uses the assigned diffusion strategy
        and that the strategy property can be updated to switch between different diffusion models.
        """
        # Initialize with GradientDiffusion strategy.
        gradient_strategy = GradientDiffusion()
        calc = DiffusionCalculator(strategy=gradient_strategy)
        xdif_grad, ydif_grad = calc.calc_diffusion(
            self.x, self.y, self.u, self.v, self.nu, self.dt
        )

        # Ensure output from gradient strategy is as expected.
        assert isinstance(xdif_grad, np.ndarray), (
            "xdif from gradient strategy should be a numpy array."
        )
        assert xdif_grad.shape == self.x.shape, "Gradient strategy xdif shape mismatch."
        assert ydif_grad.shape == self.y.shape, "Gradient strategy ydif shape mismatch."

        # Switch to RandomDiffusion strategy.
        random_strategy = RandomDiffusion()
        calc.strategy = random_strategy

        # Set a fixed seed for reproducibility.
        np.random.seed(100)
        xdif_rand, ydif_rand = calc.calc_diffusion(
            self.x, self.y, self.u, self.v, self.nu, self.dt
        )

        # Check that the output from the random strategy is a NumPy array with the expected shape.
        assert isinstance(xdif_rand, np.ndarray), (
            "xdif from random strategy should be a numpy array."
        )
        assert xdif_rand.shape == self.x.shape, "Random strategy xdif shape mismatch."
        assert ydif_rand.shape == self.y.shape, "Random strategy ydif shape mismatch."
