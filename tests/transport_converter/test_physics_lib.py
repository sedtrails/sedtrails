"""
Unit tests for physics_lib.py module.

This module comprehensively tests the physics-based calculations for sediment transport
implemented in sedtrails.transport_converter.physics_lib. The tests validate:

1. **Grain Properties**: Dimensionless grain size, critical Shields parameter, settling velocity
2. **Flow Dynamics**: Shear velocity, Shields parameter calculations
3. **Transport Mechanisms**: Bed load and suspended load velocities
4. **Layer Calculations**: Transport and mixing layer thicknesses
5. **Vector Operations**: Direction computations from transport magnitudes

Testing Strategy:
- **Known Values**: Compare against analytical solutions and literature values
- **Formula Validation**: Verify mathematical implementations match theoretical formulas
- **Range Checking**: Ensure outputs fall within physically reasonable ranges
- **Edge Cases**: Test boundary conditions (zero values, extreme parameters)
- **Integration**: Test workflows combining multiple physics functions
- **Parametrized Tests**: Systematic validation across parameter ranges

Physics Background:
The functions implement standard sediment transport theory including Shields analysis,
Soulsby formulations, and empirical relationships for particle settling and transport.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from sedtrails.transport_converter.physics_lib import (
    MixingLayerMethod,
    SuspendedVelocityMethod,
    compute_bed_load_velocity,
    compute_directions_from_magnitude,
    compute_grain_properties,
    compute_mixing_layer_thickness,
    compute_shear_velocity,
    compute_shields,
    compute_suspended_velocity,
    compute_transport_layer_thickness,
)


class TestComputeGrainProperties:
    """Test the compute_grain_properties function with known examples.

    This class validates the fundamental grain property calculations that form the basis
    of all sediment transport computations. Key properties tested:

    - **Dimensionless grain size (D*)**: Combines particle diameter with fluid properties
      Formula: D* = d * (g*(ρs/ρw - 1)/ν²)^(1/3)
    - **Critical Shields parameter (θcr)**: Threshold for particle motion initiation
    - **Settling velocity (ws)**: Terminal fall velocity in quiescent fluid
    - **Critical shear stress (τcr)**: Minimum stress needed for sediment motion

    These properties depend on grain size, density contrast, and fluid viscosity.
    Tests verify both individual calculations and physically reasonable ranges.
    """

    @pytest.fixture
    def grain_parameters(self):
        """Standard grain parameters for testing.

        These parameters represent typical marine sediment conditions:
        - Grain size: 200 μm (fine to medium sand, common in coastal environments)
        - Sediment: Quartz density (most common marine sediment mineral)
        - Fluid: Cold seawater properties (representative of many coastal areas)

        This combination yields dimensionless grain size D* ≈ 5-6, which is in the
        intermediate regime where both viscous and inertial forces are important.
        """
        return {
            'grain_diameter': 200e-6,  # 200 microns in meters - fine to medium sand
            'gravity': 9.81,  # m/s² - standard Earth gravity
            'sediment_density': 2650.0,  # kg/m³ (typical quartz sediment)
            'water_density': 1027.0,  # kg/m³ (seawater density)
            'kinematic_viscosity': 1.36e-6,  # m²/s (seawater at 10°C)
        }

    def test_settling_velocity_known_value(self, grain_parameters):
        """Test settling velocity calculation with known input/output values.

        This test validates the settling velocity calculation against a known reference value.
        The settling velocity represents the terminal fall speed of a particle in quiescent
        fluid, balancing gravitational and drag forces.

        For 200 μm quartz in cold seawater, the expected settling velocity is ~0.0202 m/s,
        calculated using Soulsby (1997, p. 136) formulations that account for particle Reynolds number
        effects across viscous, intermediate, and inertial flow regimes.

        The 1% tolerance accounts for numerical precision in iterative solutions.
        """
        result = compute_grain_properties(**grain_parameters)

        # Expected settling velocity for these parameters should be approximately 0.0202 m/s
        # This value comes from Soulsby (1997, p. 136) empirical formulation for natural sediments
        expected_settling_velocity = 0.0202

        assert_allclose(
            result['settling_velocity'],
            expected_settling_velocity,
            rtol=0.01,  # 1% relative tolerance for numerical precision
            err_msg=f'Expected settling velocity ~{expected_settling_velocity}, got {result["settling_velocity"]}',
        )

    def test_all_properties_returned(self, grain_parameters):
        """Test that all expected properties are returned."""
        result = compute_grain_properties(**grain_parameters)

        expected_keys = {'dimensionless_grain_size', 'critical_shields', 'settling_velocity', 'critical_shear_stress'}

        assert set(result.keys()) == expected_keys

        # All values should be positive and finite
        for key, value in result.items():
            assert np.isfinite(value), f'{key} should be finite, got {value}'
            assert value > 0, f'{key} should be positive, got {value}'

    def test_dimensionless_grain_size_formula(self, grain_parameters):
        """Test dimensionless grain size calculation.

        The dimensionless grain size D* is a fundamental parameter in sediment transport
        that characterizes the relative importance of gravitational vs viscous forces:

        D* = d * (g*(ρs/ρw - 1)/ν²)^(1/3)

        Where:
        - d = grain diameter
        - g = gravitational acceleration
        - ρs, ρw = sediment and water densities
        - ν = kinematic viscosity

        D* values indicate flow regime:
        - D* < 1: Viscous regime (Stokes law)
        - 1 < D* < 40: Intermediate regime (empirical formulas)
        - D* > 40: Inertial regime (Newton's law)

        This test verifies the mathematical implementation matches the theoretical formula.
        """
        result = compute_grain_properties(**grain_parameters)

        # Manual calculation: D* = ((g*(ρs/ρw - 1)/ν²)^(1/3)) * d
        # This formula combines particle and fluid properties into a single dimensionless parameter
        g = grain_parameters['gravity']
        rho_s = grain_parameters['sediment_density']
        rho_w = grain_parameters['water_density']
        nu = grain_parameters['kinematic_viscosity']
        d = grain_parameters['grain_diameter']

        # The (1/3) exponent comes from dimensional analysis balancing length scales
        expected_dstar = ((g * (rho_s / rho_w - 1) / nu**2) ** (1 / 3)) * d

        assert_allclose(
            result['dimensionless_grain_size'],
            expected_dstar,
            rtol=1e-10,  # Very tight tolerance since this is pure arithmetic
            err_msg='Dimensionless grain size calculation error',
        )

    def test_critical_shields_range(self, grain_parameters):
        """Test that critical Shields number is in reasonable range."""
        result = compute_grain_properties(**grain_parameters)

        # Critical Shields should be between 0.03 and 0.06 for typical sand
        assert 0.02 <= result['critical_shields'] <= 0.1, (
            f'Critical Shields {result["critical_shields"]} outside expected range [0.02, 0.1]'
        )

    def test_settling_velocity_range(self, grain_parameters):
        """Test that settling velocity is in reasonable range for medium sand."""
        result = compute_grain_properties(**grain_parameters)

        # For 250 micron sand, settling velocity should be around 0.02 m/s
        assert 0.01 <= result['settling_velocity'] <= 0.05, (
            f'Settling velocity {result["settling_velocity"]} outside expected range [0.01, 0.05] m/s'
        )


class TestComputeShearVelocity:
    """Test the compute_shear_velocity function.

    Shear velocity (u*) is a fundamental velocity scale in boundary layer flows:
    u* = sqrt(τ/ρ)

    Where τ is bed shear stress and ρ is fluid density. Despite its name, u* is not
    actually a velocity but rather a velocity scale that characterizes the intensity
    of turbulent mixing near the bed.

    Physical significance:
    - Controls vertical mixing in boundary layers
    - Determines sediment transport rates
    - Sets length and time scales for near-bed processes
    - Appears in logarithmic velocity profiles

    Tests validate the simple but critical square root relationship.
    """

    def test_single_value(self):
        """Test with a single value.

        This test verifies the basic shear velocity formula: u* = sqrt(τ/ρ)
        Using simple round numbers (τ = 1.0 N/m², ρ = 1000 kg/m³) makes the
        calculation transparent and easy to verify manually.

        The result (~0.0316 m/s) represents moderate turbulent conditions
        typical of coastal flows during moderate wave/current activity.
        """
        bed_shear_stress = np.array([1.0])  # N/m² - moderate shear stress
        water_density = 1000.0  # kg/m³ - fresh water density

        result = compute_shear_velocity(bed_shear_stress, water_density)
        expected = np.sqrt(1.0 / 1000.0)  # sqrt(τ/ρ) = 0.0316 m/s

        assert_allclose(result[0], expected)

    def test_array_input(self):
        """Test with array input."""
        bed_shear_stress = np.array([0.5, 1.0, 2.0])
        water_density = 1025.0

        result = compute_shear_velocity(bed_shear_stress, water_density)
        expected = np.sqrt(bed_shear_stress / water_density)

        assert_allclose(result, expected)

    def test_negative_shear_stress(self):
        """Test handling of negative shear stress (should use absolute value).

        In numerical models, shear stress can sometimes be computed as negative
        due to coordinate conventions or numerical artifacts. However, shear velocity
        represents the magnitude of turbulent intensity and must always be positive.

        The function should take the absolute value of shear stress before computing
        the square root. This ensures physical consistency while being robust to
        sign conventions in different numerical implementations.
        """
        bed_shear_stress = np.array([-1.0])  # Negative stress (coordinate convention artifact)
        water_density = 1000.0

        result = compute_shear_velocity(bed_shear_stress, water_density)
        expected = np.sqrt(1.0 / 1000.0)  # Should be same as positive value

        assert_allclose(result[0], expected)

    def test_zero_shear_stress(self):
        """Test with zero shear stress."""
        result = compute_shear_velocity(np.array([0.0]), 1000.0)
        assert result[0] == 0.0


class TestComputeShields:
    """Test the compute_shields function.

    The Shields parameter (θ) is the fundamental dimensionless parameter for
    sediment transport, representing the ratio of destabilizing (flow) to
    stabilizing (gravitational) forces acting on bed particles:

    θ = τ / (g*(ρs - ρw)*d)

    Where:
    - τ = bed shear stress
    - g = gravitational acceleration
    - ρs, ρw = sediment and water densities
    - d = grain diameter

    Physical interpretation:
    - θ < θcr: No sediment motion (stable bed)
    - θ ≈ θcr: Incipient motion (threshold conditions)
    - θ > θcr: Active transport (increasing with θ)

    The critical Shields parameter θcr ≈ 0.03-0.06 for most natural sediments.
    """

    @pytest.fixture
    def shields_params(self):
        """Standard parameters for Shields calculation."""
        return {
            'gravity': 9.81,
            'sediment_density': 2650.0,
            'water_density': 1025.0,
            'grain_diameter': 250e-6,
        }

    def test_formula_implementation(self, shields_params):
        """Test that the Shields formula is correctly implemented.

        This test verifies the mathematical implementation of the Shields parameter
        formula by comparing against manual calculation. The denominator represents
        the immersed weight per unit area of a grain:

        g*(ρs - ρw)*d = specific weight difference × grain diameter

        This creates the proper force balance:
        - Numerator (τ): Hydrodynamic force trying to move the particle
        - Denominator: Gravitational force (reduced by buoyancy) resisting motion

        The result is dimensionless, allowing universal comparison across different
        sediment types and flow conditions.
        """
        bed_shear_stress = np.array([1.0])  # N/m² - applied hydrodynamic stress

        result = compute_shields(bed_shear_stress, **shields_params)

        # Manual calculation: θ = τ / (g(ρs - ρw)d)
        # Denominator is the immersed weight per unit area of a grain layer
        expected = bed_shear_stress[0] / (
            shields_params['gravity']
            * (shields_params['sediment_density'] - shields_params['water_density'])
            * shields_params['grain_diameter']
        )

        assert_allclose(result[0], expected)

    def test_array_input(self, shields_params):
        """Test with array input."""
        bed_shear_stress = np.array([0.5, 1.0, 2.0])

        result = compute_shields(bed_shear_stress, **shields_params)

        assert isinstance(result, np.ndarray)
        assert result.shape == bed_shear_stress.shape
        assert np.all(result >= 0)


class TestComputeBedLoadVelocity:
    """Test the compute_bed_load_velocity function.

    Bed load transport occurs when particles roll, slide, and hop along the bed
    while maintaining frequent contact. The bed load velocity represents the
    average speed of this near-bed particle motion.

    Key physics:
    - Only occurs above critical Shields parameter (θ > θcr)
    - Velocity increases with excess shear stress above threshold
    - Proportional to shear velocity with empirical corrections
    - Much slower than flow velocity due to particle-bed interactions

    The implemented formula follows Soulsby (1997):
    U_bed = 10 * u* * (1 - 0.7 * sqrt(θcr / θ))

    Where the factor 10 and correction term are empirically determined (see Fredsoe & Deigaard, 1992, eq 7.51).
    """

    def test_above_critical_conditions(self):
        """Test bed load velocity above critical conditions."""
        shields_number = np.array([0.1])
        critical_shields = 0.05
        mean_shear_velocity = np.array([0.01])

        result = compute_bed_load_velocity(shields_number, critical_shields, mean_shear_velocity)

        # Should return positive velocity
        assert result[0] > 0
        assert np.isfinite(result[0])

    def test_below_critical_conditions(self):
        """Test bed load velocity below critical conditions."""
        shields_number = np.array([0.03])
        critical_shields = 0.05
        mean_shear_velocity = np.array([0.01])

        result = compute_bed_load_velocity(shields_number, critical_shields, mean_shear_velocity)

        # Should return zero velocity
        assert result[0] == 0.0

    def test_formula_implementation(self):
        """Test the bed load velocity formula implementation.

        This test verifies the Soulsby (1997) empirical formula for bed load velocity:
        U_bed = 10 * u* * (1 - 0.7 * sqrt(θcr / θ))

        Formula components:
        - Factor "10": Empirical coefficient relating particle to shear velocity
        - Correction term: Accounts for threshold effects (approaches 0 as θ → θcr)
        - u*: Provides velocity scale proportional to flow intensity

        Physical interpretation:
        - At threshold (θ = θcr): correction = 1 - 0.7 = 0.3, giving U_bed = 3*u*
        - For strong flows (θ >> θcr): correction → 1, giving U_bed → 10*u*

        The square root dependency reflects the nonlinear response to excess stress.
        """
        shields_number = np.array([0.1])  # Well above critical (strong transport)
        critical_shields = 0.05  # Typical critical value for sand
        mean_shear_velocity = np.array([0.02])  # Moderate shear velocity (m/s)

        result = compute_bed_load_velocity(shields_number, critical_shields, mean_shear_velocity)

        # Manual calculation: U_bed = 10 * u*_mean * (1 - 0.7 * sqrt(θ_cr / θ_max))
        # The correction factor accounts for threshold effects in particle motion
        expected = 10.0 * mean_shear_velocity[0] * (1 - 0.7 * np.sqrt(critical_shields / shields_number[0]))

        assert_allclose(result[0], expected)


class TestComputeTransportLayerThickness:
    """Test the compute_transport_layer_thickness function.

    The transport layer thickness represents the vertical extent of the active
    sediment layer where particles are being transported. This is computed from
    mass conservation principles:

    Mass flux = Volume flux × Bulk density
    Q_mass = Q_vol × ρs × (1 - n)

    Where:
    - Q_mass = sediment mass flux (kg/m/s)
    - Q_vol = sediment volume flux (m²/s)
    - ρs = sediment grain density
    - n = porosity (void fraction)

    The layer thickness is then: d = Q_vol / U = Q_mass / (U × ρs × (1-n))

    This thickness is used in advection-diffusion models to represent the
    vertical scale of active sediment transport processes.
    """

    def test_nonzero_velocity(self):
        """Test with non-zero velocity.

        This test validates the mass conservation calculation for transport layer thickness.
        The calculation converts mass flux to volume flux by dividing by bulk density:

        Steps:
        1. Convert mass flux (kg/m/s) to volume flux (m²/s) using bulk density
        2. Bulk density = ρs × (1-n) accounts for grain density and packing
        3. Layer thickness = volume flux / velocity (continuity equation)

        Physical interpretation:
        - Higher transport rate → thicker active layer
        - Higher velocity → thinner layer (same mass spread over less thickness)
        - Higher density/lower porosity → thinner layer (more compact packing)

        Typical values: thickness ~mm to cm for natural sediment transport.
        """
        transport_magnitude = np.array([1.0])  # kg/m/s - mass flux per unit width
        velocity_magnitude = np.array([0.5])  # m/s - particle transport velocity
        sediment_density = 2650.0  # kg/m³ - quartz grain density
        porosity = 0.4  # 40% void space (typical for sand)

        result = compute_transport_layer_thickness(transport_magnitude, velocity_magnitude, sediment_density, porosity)

        # Manual calculation: d_layer = (Q_volume) / U_layer
        # Convert mass flux to volume flux: Q_volume = Q_mass / (ρ_s * (1 - n))
        # Bulk density ρ_bulk = ρ_s * (1-n) accounts for particle packing
        transport_flux = transport_magnitude[0] / (sediment_density * (1 - porosity))
        expected = transport_flux / velocity_magnitude[0]  # Continuity: thickness = flux/velocity

        assert_allclose(result[0], expected)

    def test_zero_velocity(self):
        """Test with zero velocity."""
        transport_magnitude = np.array([1.0])
        velocity_magnitude = np.array([0.0])
        sediment_density = 2650.0
        porosity = 0.4

        result = compute_transport_layer_thickness(transport_magnitude, velocity_magnitude, sediment_density, porosity)

        assert result[0] == 0.0


class TestComputeSuspendedVelocity:
    """Test the compute_suspended_velocity function.

    Suspended sediment transport occurs when particles are lifted into the water
    column by turbulent mixing and advected by the flow. Key physics:

    **Suspension mechanism**:
    - Turbulent eddies lift particles against gravity
    - Balance between upward turbulent mixing and downward settling
    - Requires shear velocity u* > settling velocity ws for strong suspension

    **Velocity calculation**:
    - Suspended particles move at ~flow velocity (high coupling)
    - Corrections for settling lag and concentration profiles
    - Method-dependent formulations (Soulsby 2011, etc.)

    **Critical conditions**:
    - Below threshold: no suspension (ws dominates)
    - Above threshold: velocity ≈ flow velocity with empirical corrections

    Tests validate method implementations and threshold behavior.
    """

    @pytest.fixture
    def suspended_params(self):
        """Standard parameters for suspended velocity calculation."""
        return {
            'flow_velocity_magnitude': np.array([0.5]),
            'bed_load_velocity': np.array([0.1]),
            'settling_velocity': 0.02,
            'von_karman_constant': 0.4,
            'max_shear_velocity': np.array([0.03]),
            'shields_number': np.array([0.1]),
            'critical_shields': 0.05,
            'method': SuspendedVelocityMethod.SOULSBY_2011,
        }

    def test_above_critical_conditions(self, suspended_params):
        """Test suspended velocity above critical conditions."""
        result = compute_suspended_velocity(**suspended_params)

        # Should return positive velocity
        assert result[0] > 0
        assert np.isfinite(result[0])

    def test_below_critical_conditions(self, suspended_params):
        """Test suspended velocity below critical conditions."""
        suspended_params['shields_number'] = np.array([0.03])  # Below critical

        result = compute_suspended_velocity(**suspended_params)

        # Should return zero velocity
        assert result[0] == 0.0

    def test_invalid_method(self, suspended_params):
        """Test error handling for invalid method."""
        suspended_params['method'] = 'invalid_method'

        with pytest.raises(ValueError, match='Unknown suspended velocity method'):
            compute_suspended_velocity(**suspended_params)

    def test_macdonald_method_matches_centroid_log_profile(self, suspended_params):
        """MacDonald velocity should match the centroid-height log profile."""
        max_shear_velocity = np.array([0.04, 0.08])
        water_depth = np.array([2.0, 3.0])
        settling_velocity = 0.02
        von_karman_constant = 0.4
        grain_diameter = 2.5e-4

        result = compute_suspended_velocity(
            flow_velocity_magnitude=np.array([0.5, 0.8]),
            bed_load_velocity=np.array([0.1, 0.2]),
            settling_velocity=settling_velocity,
            von_karman_constant=von_karman_constant,
            max_shear_velocity=max_shear_velocity,
            shields_number=np.array([0.1, 0.2]),
            critical_shields=0.05,
            method=SuspendedVelocityMethod.MACDONALD_2006,
            water_depth=water_depth,
            grain_diameter=grain_diameter,
        )

        ws_ratio = settling_velocity / (von_karman_constant * max_shear_velocity)
        z_s = water_depth * 0.0398 * np.power(10.0, -1.08 * np.tanh(1.2 * np.log(ws_ratio) - 0.4))
        k_s = 2.5 * grain_diameter
        expected = np.maximum(2.5 * max_shear_velocity * np.log(30.0 * z_s / k_s), 0.0)
        assert_allclose(result, expected)

    def test_macdonald_method_requires_depth_and_grain_size(self, suspended_params):
        """MacDonald velocity requires water depth and grain diameter inputs."""
        suspended_params['method'] = SuspendedVelocityMethod.MACDONALD_2006

        with pytest.raises(ValueError, match='water_depth.*grain_diameter'):
            compute_suspended_velocity(**suspended_params)

    def test_macdonald_method_returns_zero_for_zero_shear(self, suspended_params):
        """Zero shear velocity should not emit NaN or negative MacDonald speeds."""
        suspended_params.update(
            {
                'method': SuspendedVelocityMethod.MACDONALD_2006,
                'max_shear_velocity': np.array([0.0]),
                'water_depth': np.array([2.0]),
                'grain_diameter': 2.5e-4,
            }
        )

        result = compute_suspended_velocity(**suspended_params)

        assert result[0] == 0.0


class TestComputeDirectionsFromMagnitude:
    """Test the compute_directions_from_magnitude function.

    This function performs vector operations to compute velocity components from
    transport information. The key concept is directional scaling:

    **Vector relationship**:
    - Transport vector: T = (Tx, Ty) with magnitude |T|
    - Unit direction: û = T/|T| = (Tx/|T|, Ty/|T|)
    - Velocity vector: V = |V| * û = |V| * (Tx/|T|, Ty/|T|)

    **Physical interpretation**:
    - Assumes velocity direction follows transport direction
    - Scales velocity magnitude by transport direction ratios
    - Handles zero transport (no preferred direction) → zero velocity

    **Applications**:
    - Converting scalar velocity magnitudes to vector fields
    - Maintaining consistent directional relationships in transport models
    """

    def test_direction_calculation(self):
        """Test direction calculation from magnitude and transport components.

        This test uses a classic 3-4-5 right triangle to verify vector scaling:
        - Transport vector: (3, 4) with magnitude 5
        - Unit direction vector: (3/5, 4/5) = (0.6, 0.8)
        - Velocity magnitude: 2.0
        - Expected velocity: 2.0 * (0.6, 0.8) = (1.2, 1.6)

        The calculation preserves the transport direction while scaling to the
        specified velocity magnitude. This is equivalent to:
        V = |V| * (T/|T|) = |V| * unit_direction_vector

        Verification: |V_result| = sqrt(1.2² + 1.6²) = 2.0 ✓
        """
        velocity_magnitude = np.array([2.0])  # Desired velocity magnitude
        transport_x = np.array([3.0])  # Transport x-component
        transport_y = np.array([4.0])  # Transport y-component
        transport_magnitude = np.array([5.0])  # |T| = sqrt(3² + 4²) = 5

        velocity_x, velocity_y = compute_directions_from_magnitude(
            velocity_magnitude, transport_x, transport_y, transport_magnitude
        )

        # Expected: V = |V| * (T/|T|) - vector scaling by unit direction
        expected_x = velocity_magnitude[0] * (transport_x[0] / transport_magnitude[0])  # 2.0 * (3/5) = 1.2
        expected_y = velocity_magnitude[0] * (transport_y[0] / transport_magnitude[0])  # 2.0 * (4/5) = 1.6

        assert_allclose(velocity_x[0], expected_x)
        assert_allclose(velocity_y[0], expected_y)

    def test_zero_transport_magnitude(self):
        """Test with zero transport magnitude."""
        velocity_magnitude = np.array([2.0])
        transport_x = np.array([0.0])
        transport_y = np.array([0.0])
        transport_magnitude = np.array([0.0])

        velocity_x, velocity_y = compute_directions_from_magnitude(
            velocity_magnitude, transport_x, transport_y, transport_magnitude
        )

        assert velocity_x[0] == 0.0
        assert velocity_y[0] == 0.0

    def test_magnitude_consistency(self):
        """Test that computed velocity components have correct magnitude."""
        velocity_magnitude = np.array([1.0])
        transport_x = np.array([0.6])
        transport_y = np.array([0.8])
        transport_magnitude = np.array([1.0])  # Unit vector

        velocity_x, velocity_y = compute_directions_from_magnitude(
            velocity_magnitude, transport_x, transport_y, transport_magnitude
        )

        computed_magnitude = np.sqrt(velocity_x[0] ** 2 + velocity_y[0] ** 2)
        assert_allclose(computed_magnitude, velocity_magnitude[0])


class TestComputeMixingLayerThickness:
    """Test the compute_mixing_layer_thickness function.

    The mixing layer represents the vertical extent of active sediment-water mixing
    near the bed. This layer controls:

    **Physical processes**:
    - Vertical diffusion of suspended sediment
    - Exchange between bed and water column
    - Concentration profile development
    - Particle residence times

    **Bertin (2008) formulation**:
    d_mix = 0.041 * sqrt(max(τ_max - τ_cr, 0))

    Where:
    - Coefficient 0.041 is empirically determined
    - Square root dependency reflects turbulent scaling
    - Only excess stress above critical threshold contributes
    - Units: meters (typical range: mm to cm)

    **Applications**:
    - Boundary conditions for suspension models
    - Scaling vertical diffusion coefficients
    - Estimating near-bed concentration gradients
    """

    def test_bertin_method(self):
        """Test Bertin (2008) method.

        The Bertin (2008) formulation relates mixing layer thickness to excess
        shear stress above the critical threshold:

        d_mix = 0.041 * sqrt(τ_max - τ_cr)

        **Physical basis**:
        - Only excess stress above critical contributes to mixing
        - Square root scaling follows turbulent boundary layer theory
        - Coefficient 0.041 m/(N/m²)^0.5 from field measurements

        **Example calculation**:
        - τ_max = 2.0 N/m² (strong flow conditions)
        - τ_cr = 1.0 N/m² (critical threshold)
        - Excess stress = 1.0 N/m²
        - Expected thickness = 0.041 * sqrt(1.0) = 0.041 m = 4.1 cm

        This represents a substantial mixing layer typical of energetic coastal flows.
        """
        max_bed_shear_stress = np.array([2.0])  # N/m² - strong flow conditions
        critical_shear_stress = 1.0  # N/m² - sediment motion threshold

        result = compute_mixing_layer_thickness(
            max_bed_shear_stress, critical_shear_stress, method=MixingLayerMethod.BERTIN_2008
        )

        # Manual calculation: d_mix = 0.041 * sqrt(max(τ_max - τ_cr, 0))
        # Only excess stress above threshold contributes to mixing layer development
        expected = 0.041 * np.sqrt(max_bed_shear_stress[0] - critical_shear_stress)

        assert_allclose(result[0], expected)

    def test_bertin_method_uses_custom_coefficient(self):
        """A custom Bertin coefficient should scale the mixing depth."""
        max_bed_shear_stress = np.array([5.0])
        critical_shear_stress = 1.0

        result = compute_mixing_layer_thickness(
            max_bed_shear_stress,
            critical_shear_stress,
            method=MixingLayerMethod.BERTIN_2008,
            bertin_coefficient=0.08,
        )

        assert_allclose(result[0], 0.08 * np.sqrt(4.0))

    def test_bertin_method_digitized_values(self):
        """Test Bertin (2008) method against digitized values from Figure 3.

        This test validates the Bertin (2008) formulation using data points
        digitized from Figure 3 in the original paper. The figure shows the
        relationship between excess shear stress and mixing layer thickness
        for various field conditions.

        **Data source**: Bertin et al. (2008), Figure 3
        **Formula**: d_mix = 0.041 * sqrt(τ_max - τ_cr)

        The test uses representative data points that span the range of
        conditions presented in the paper, from moderate to high energy
        coastal environments.
        """
        # Critical shear stress (assumed constant for this test)
        # based on 200 um sand in seawater
        # (Bertin et al. (2008) use 0.18-0.22 mm sand)
        critical_shear_stress = 0.176  # N/m² -

        # Digitized data pairs from Bertin et al. (2008) Figure 3
        # Format: (max_bed_shear_stress [N/m²], expected_d_mix [m])
        test_data = [
            (0.27, 0.015),
            (0.8, 0.032),
            (1.65, 0.05),
        ]

        for max_stress, expected_thickness in test_data:
            max_bed_shear_stress = np.array([max_stress])

            result = compute_mixing_layer_thickness(
                max_bed_shear_stress, critical_shear_stress, method=MixingLayerMethod.BERTIN_2008
            )

            # Test against digitized values with reasonable tolerance
            # Allow for digitization uncertainty and model assumptions
            assert_allclose(
                result[0],
                expected_thickness,
                rtol=0.20,  # 20% relative tolerance for digitization uncertainty
                err_msg=f'Failed for τ_max={max_stress} N/m², expected d_mix≈{expected_thickness} m',
            )

    def test_below_critical_stress(self):
        """Test with shear stress below critical value."""
        max_bed_shear_stress = np.array([0.5])
        critical_shear_stress = 1.0

        result = compute_mixing_layer_thickness(
            max_bed_shear_stress, critical_shear_stress, method=MixingLayerMethod.BERTIN_2008
        )

        assert result[0] == 0.0

    def test_harris_wiberg_method_not_implemented(self):
        """Test that Harris-Wiberg method raises NotImplementedError."""
        max_bed_shear_stress = np.array([2.0])
        critical_shear_stress = 1.0

        with pytest.raises(NotImplementedError):
            compute_mixing_layer_thickness(
                max_bed_shear_stress, critical_shear_stress, method=MixingLayerMethod.HARRIS_WIBERG
            )

    def test_invalid_method(self):
        """Test error handling for invalid method."""
        max_bed_shear_stress = np.array([2.0])
        critical_shear_stress = 1.0

        with pytest.raises(ValueError, match='Unknown mixing layer method'):
            # Use getattr to bypass type checking for invalid method
            invalid_method = getattr(MixingLayerMethod, 'INVALID', 'invalid_method')
            compute_mixing_layer_thickness(
                max_bed_shear_stress,
                critical_shear_stress,
                method=invalid_method,  # type: ignore
            )


class TestPhysicsLibIntegration:
    """Integration tests combining multiple functions.

    These tests validate the complete sediment transport calculation workflow,
    ensuring that individual physics functions work together correctly:

    **Typical calculation sequence**:
    1. Grain properties → fundamental particle characteristics
    2. Flow conditions → shear stress and velocity fields
    3. Dimensionless parameters → Shields number, transport thresholds
    4. Transport rates → bed load and suspended load calculations
    5. Layer properties → transport and mixing layer thicknesses

    **Integration validation**:
    - Consistent units and scaling across functions
    - Physical parameter ranges maintained through workflow
    - Proper threshold behavior at critical conditions
    - Sensitivity to key parameters (grain size, flow strength)

    These tests simulate realistic sediment transport modeling scenarios.
    """

    def test_grain_properties_to_transport_workflow(self):
        """Test a complete workflow from grain properties to transport calculations.

        This integration test simulates a realistic sediment transport calculation
        workflow, starting from basic particle properties and building up to
        transport velocities. This mirrors the typical sequence in numerical models:

        **Workflow steps**:
        1. **Grain characterization**: Compute D*, θcr, ws from particle properties
        2. **Flow analysis**: Convert bed stress to dimensionless parameters
        3. **Threshold assessment**: Compare θ to θcr for transport activation
        4. **Transport calculation**: Compute particle velocities if θ > θcr

        **Physical consistency checks**:
        - All parameters remain finite and physically reasonable
        - Proper scaling relationships maintained
        - Transport only occurs above critical conditions

        This represents conditions for 250 μm sand in moderate coastal flows.
        """
        # Step 1: Compute fundamental grain properties for medium sand
        grain_params = {
            'grain_diameter': 250e-6,  # 250 μm - medium sand
            'gravity': 9.81,  # Standard gravity
            'sediment_density': 2650.0,  # Quartz density (kg/m³)
            'water_density': 1025.0,  # Seawater density (kg/m³)
            'kinematic_viscosity': 1.36e-6,  # Cold seawater viscosity (m²/s)
        }

        # Calculate particle characteristics (D*, θcr, ws, τcr)
        grain_props = compute_grain_properties(**grain_params)

        # Step 2: Apply representative flow conditions
        bed_shear_stress = np.array([1.0])  # N/m² - moderate coastal flow

        # Step 3: Compute dimensionless flow parameters
        # Shields parameter: θ = τ/(g*(ρs-ρw)*d)
        shields = compute_shields(
            bed_shear_stress,
            grain_params['gravity'],
            grain_params['sediment_density'],
            grain_params['water_density'],
            grain_params['grain_diameter'],
        )

        # Shear velocity: u* = sqrt(τ/ρw)
        shear_velocity = compute_shear_velocity(bed_shear_stress, grain_params['water_density'])

        # Step 4: Calculate transport velocities (if above threshold)
        bed_load_vel = compute_bed_load_velocity(shields, grain_props['critical_shields'], shear_velocity)

        # Validate physical consistency across the workflow
        assert np.isfinite(shields[0]), 'Shields parameter should be finite'
        assert np.isfinite(shear_velocity[0]), 'Shear velocity should be finite'
        assert np.isfinite(bed_load_vel[0]), 'Bed load velocity should be finite'
        assert shields[0] >= 0, 'Shields parameter should be non-negative'
        assert shear_velocity[0] >= 0, 'Shear velocity should be non-negative'
        assert bed_load_vel[0] >= 0, 'Bed load velocity should be non-negative'

    def test_parameter_sensitivity(self):
        """Test that functions respond sensibly to parameter changes.

        This test validates the expected physical relationships between grain size
        and transport properties. Key physics principles being tested:

        **Settling velocity vs grain size**:
        - Larger particles fall faster due to increased gravitational force
        - Relationship is nonlinear due to changing drag regimes
        - For sand-sized particles: ws ~ d^1.5 to d^2 (empirically)

        **Critical shear stress vs grain size**:
        - Larger particles require more force to initiate motion
        - Relationship: τcr = θcr * g * (ρs-ρw) * d
        - Since θcr varies slowly with size, τcr ≈ proportional to d

        **Test methodology**:
        - Use 2:1 size ratio (125 μm : 250 μm : 500 μm)
        - Covers fine to coarse sand range
        - Validates monotonic relationships expected from theory
        """
        base_params = {
            'grain_diameter': 250e-6,  # Reference: medium sand
            'gravity': 9.81,
            'sediment_density': 2650.0,
            'water_density': 1025.0,
            'kinematic_viscosity': 1.36e-6,
        }

        # Test grain size sensitivity across realistic sand size range
        fine_params = base_params.copy()
        fine_params['grain_diameter'] = 125e-6  # Fine sand (half size)

        coarse_params = base_params.copy()
        coarse_params['grain_diameter'] = 500e-6  # Coarse sand (double size)

        # Calculate properties for each size class
        base_props = compute_grain_properties(**base_params)
        fine_props = compute_grain_properties(**fine_params)
        coarse_props = compute_grain_properties(**coarse_params)

        # Validate expected physical relationships

        # Settling velocity should increase monotonically with grain size
        # Physics: larger particles have higher terminal velocity
        assert fine_props['settling_velocity'] < base_props['settling_velocity'] < coarse_props['settling_velocity']

        # Critical shear stress should increase monotonically with grain size
        # Physics: larger particles require more force to move
        assert (
            fine_props['critical_shear_stress']
            < base_props['critical_shear_stress']
            < coarse_props['critical_shear_stress']
        )


# Parametrized tests for edge cases
class TestEdgeCases:
    """Test edge cases and boundary conditions.

    Edge case testing is critical for numerical robustness in sediment transport
    models, which must handle extreme conditions that may occur in nature:

    **Why edge cases matter**:
    - Models encounter wide parameter ranges in real applications
    - Numerical instabilities often appear at extremes
    - Physical laws must remain valid across all scales
    - Graceful degradation prevents model crashes

    **Testing strategy**:
    - **Zero values**: Test mathematical limits and boundary behavior
    - **Very small values**: Ensure numerical precision is maintained
    - **Very large values**: Verify no overflow or unrealistic results
    - **Physical limits**: Test across realistic environmental ranges

    **Parametrized approach**:
    Uses pytest.mark.parametrize to systematically test multiple values,
    ensuring comprehensive coverage without code duplication.
    """

    @pytest.mark.parametrize('shear_stress', [0.0, 1e-10, 1e10])
    def test_shear_velocity_edge_values(self, shear_stress):
        """Test shear velocity with edge values.

        This parametrized test covers the full range of possible shear stress values:

        - **Zero stress (0.0)**: Calm conditions, u* should be exactly 0
        - **Tiny stress (1e-10)**: Near-zero but finite, tests numerical precision
        - **Huge stress (1e10)**: Extreme conditions (hurricane/tsunami), tests overflow

        **Physical interpretation**:
        - 1e-10 N/m²: Virtually stagnant water
        - 1e10 N/m²: Catastrophic flow conditions (u* ≈ 100 m/s)

        **Robustness requirements**:
        - Results must always be finite (no NaN/Inf)
        - Results must be non-negative (physical constraint)
        - Square root operation must handle full range safely
        """
        result = compute_shear_velocity(np.array([shear_stress]), 1000.0)
        assert np.isfinite(result[0]), f'Shear velocity should be finite for stress {shear_stress}'
        assert result[0] >= 0, f'Shear velocity should be non-negative for stress {shear_stress}'

    @pytest.mark.parametrize('grain_size', [1e-6, 1e-3, 1e-2])  # 1 micron to 1 cm
    def test_grain_properties_size_range(self, grain_size):
        """Test grain properties across realistic size range.

        This test spans the full range of natural sediment sizes encountered
        in marine and coastal environments:

        **Size classes tested**:
        - **1 μm (1e-6 m)**: Clay particles, cohesive sediments
        - **1 mm (1e-3 m)**: Coarse sand to fine gravel
        - **1 cm (1e-2 m)**: Gravel, shells, coarse bed material

        **Physics across scales**:
        - Clay: Viscous forces dominate (D* < 1), slow settling
        - Sand: Intermediate regime (1 < D* < 40), complex drag
        - Gravel: Inertial forces dominate (D* > 40), fast settling

        **Validation criteria**:
        - All properties must remain positive and finite
        - Formulations must handle different flow regimes correctly
        - No mathematical singularities or unrealistic values
        """
        params = {
            'grain_diameter': grain_size,
            'gravity': 9.81,
            'sediment_density': 2650.0,
            'water_density': 1025.0,
            'kinematic_viscosity': 1.36e-6,
        }

        result = compute_grain_properties(**params)

        # Validate mathematical and physical constraints across all size scales
        for property_name, value in result.items():
            assert np.isfinite(value), f'{property_name} should be finite for grain size {grain_size * 1e6:.1f} μm'
            assert value > 0, f'{property_name} should be positive for grain size {grain_size * 1e6:.1f} μm'

    @pytest.mark.parametrize('velocity_mag', [0.0, 1e-6, 1.0, 10.0])
    def test_transport_layer_thickness_velocities(self, velocity_mag):
        """Test transport layer thickness with different velocities.

        This test examines transport layer behavior across the full range of
        possible particle velocities, from stagnant to extreme conditions:

        **Velocity scenarios**:
        - **0.0 m/s**: Stagnant conditions, thickness should be 0 (special case)
        - **1e-6 m/s**: Nearly stagnant, very thick layer (extreme case)
        - **1.0 m/s**: Typical transport velocity in energetic flows
        - **10.0 m/s**: Extreme velocity (storm/tsunami conditions)

        **Physical expectations**:
        - Layer thickness = volume flux / velocity (continuity)
        - Higher velocity → thinner layer (same mass, faster transport)
        - Zero velocity should be handled gracefully (avoid division by zero)

        **Robustness testing**:
        - Function should handle full velocity range without errors
        - Results must be physically meaningful (finite, non-negative)
        """
        transport_mag = np.array([1.0])  # Fixed mass flux (kg/m/s)
        velocity_magnitude = np.array([velocity_mag])

        result = compute_transport_layer_thickness(transport_mag, velocity_magnitude, 2650.0, 0.4)

        # Validate mathematical and physical constraints
        assert np.isfinite(result[0]), f'Layer thickness should be finite for velocity {velocity_mag} m/s'
        assert result[0] >= 0, f'Layer thickness should be non-negative for velocity {velocity_mag} m/s'


if __name__ == '__main__':
    """
    Direct test execution capability.
    
    This allows the test file to be run directly with:
    python test_physics_lib.py
    
    The "-v" flag enables verbose output showing individual test names and results.
    This is useful for:
    - Quick validation during development
    - Debugging specific test failures  
    - Understanding test coverage and organization
    - Running tests without pytest discovery overhead
    """
    # Run tests when script is executed directly with verbose output
    pytest.main([__file__, '-v'])
