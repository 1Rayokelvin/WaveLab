# WaveLab

A Python library for Optical field simulations in vacuum using plane wave expansion and topological singularity analysis.

WaveLab computes fully 3D fields (electric and magnetic fields, along with spatial derivatives) using an angular spectrum decomposition. The plane-wave expansion is evaluated via robust solid-angle discretization (Fibonacci sphere sampling), enabling the construction of a wide variety of physically rigorous fields (e.g., Laguerre-Gauss, Speckles, Polychromatic). 

The package also includes numerical tools to detect and trace polarization singularities (C-points, L-lines) in three dimensions.

## Requirements
- `numpy`
- `scipy`
- `numba` (Optional, but highly recommended for computational speed)
- `pyvista` (Optional, for 3D k-space visualization)
- `matplotlib` (For plotting)

---

## Physics & Conventions

To ensure physical rigor and numerical stability, WaveLab employs the following conventions:

1. **Natural Units:** Computations use dimensionless natural units where $c = 1$ and $\epsilon_0 = \mu_0 = 1$. Spatial units (`x, y, z`, `op.spacing`) are defined entirely by the user's choice of `wavelength` units. Time `t` is treated as proper time ($ct$) and shares spatial units.
2. **Phase Convention:** WaveLab uses the standard optics phase convention: $E(\mathbf{r}, t) = \sum \mathbf{c}_n e^{i(\mathbf{k}_n \cdot \mathbf{r} - \omega_n t)}$.
3. **3D Polarization Basis:** When generating plane waves, the 2D Jones vector `config.source.pol_vect` is mapped to the 3D unit sphere. This is done by creating a local transverse coordinate frame for *each* $k$-vector using a Rodrigues rotation (parallel transport) from the main beam axis.
4. **Custom Profiles:** When providing custom `Callable` functions to `k_space.custom()` or `polychromatic.custom()`, WaveLab will test them with dummy variables during configuration to validate tensor shapes and types.

---

## Library Workflow

The computation pipeline is strictly divided into three stages:
1. **Configuration (`Config`)**: Define physical parameters (grid size, wavelength, spatial/spectral profiles, randomness).
2. **Generation (`BeamMaker`)**: Precompute the discrete set of plane waves (wavevectors, complex vector amplitudes) into a lightweight `Beam` object.
3. **Execution (`FieldEngine`)**: Route the `Beam` to a backend (`numba` or `numpy`) to evaluate the fields on 2D grids, 3D point clouds, or individual points, returning a unified `FieldResult`.

---

## Basic Usage

### 1. Simulating a Polychromatic Optical Field
This example generates the intensity of a Gaussian spatial profile with a Lorentzian spectral envelope.

```python
import numpy as np
import matplotlib.pyplot as plt
from wavelab import get_config, BeamMaker, FieldEngine

# 1. Define configuration
config = get_config()
config.op.size = (5.0, 5.0)          # 5x5 Observation Plane
config.op.spacing = 0.05
config.source.num_modes = 15000      # Number of plane waves

# Wavelengths around a center
config.source.wavelength = np.linspace(0.2, 0.8, 11).tolist()

# Define spectral and spatial distributions via fluent API
config.source.k_space.gaussian(sigma_k_perp=1.5)
config.source.polychromatic.lorentzian(center=0.5, gamma=0.1)

# Turn off stochastic effects for a deterministic beam
config.source.randomize.off()

# 2. Precompute the Beam and initialize the Engine
beam = BeamMaker(config).generate_beam()
engine = FieldEngine(beam, config)

# Optional: View the physics summary of the generated beam
beam.summary()

# 3. Compute fields on the observation plane
result = engine.compute_on_op(z=0.0, t=0.0)

# Plot Intensity
plt.imshow(result.intensity_E, cmap='inferno', extent=engine.op_extent, origin='lower')
plt.title('Field Intensity')
plt.colorbar(label='|E|²')
plt.show()
```

### 2. Finding C-points in a Speckle Field
This example generates the Stokes parameters of a speckle field and locates C-points.

```
import matplotlib.pyplot as plt
from wavelab import get_config, BeamMaker, FieldEngine
from wavelab.utils import get_stokes_params
from wavelab.singularities import SingularityFinder

# 1. Define configuration
config = get_config()
config.op.size = (5.0, 5.0)      
config.source.num_modes = 15000  
config.source.wavelength = 1.0   

# Define spatial distributions
config.source.k_space.gaussian(sigma_k_perp=1.5)

# Keep stochastic effects (RandomizeConfig defaults to True -> creates speckles!)

# 2. Setup Beam and Engine
beam = BeamMaker(config).generate_beam()
engine = FieldEngine(beam, config)

# 3. Compute fields on the observation plane
result = engine.compute_on_op(z=0.0, t=0.0, need_b=False, need_derivs=False)

# Get Stokes parameters (Using Ex and Ey by default)
# result.E shape is (3, Ny, Nx). E[0] is Ex, E[1] is Ey.
s_all = get_stokes_params(result.E[0], result.E[1], normalize=True) 

# 4. Find C points inside the observation plane
finder = SingularityFinder(engine)
c_points = finder.find_stokes_C_points(z_value=0.0, E_grid=result.E)

# Extract positions and handedness from confident data
cp_positions = [pt['position'] for pt in c_points if pt['confident']]
cp_handedness = [pt['handedness'] for pt in c_points if pt['confident']]

if cp_positions:
    x, y, z = zip(*cp_positions)
    plt.scatter(x, y, c=cp_handedness, cmap='bwr', edgecolor='k', zorder=5)

# Plot s3 Field and C points
plt.imshow(s_all['s3'], cmap='coolwarm', origin='lower', extent=engine.op_extent)
plt.colorbar(label='Normalized s3')
plt.title('s3 field with C-points marked')
plt.show()
```