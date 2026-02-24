# WaveLab

A Python library for Optical field simulations in vacuum using plane wave expansion and topological singularity analysis.

WaveLab computes fully 3D fields (electric and magnetic fields, along with spatial derivatives) using angular spectrum decomposition. The plane-wave expansion is evaluated via solid-angle discretization over the unit sphere, enabling construction of variety of fields (e.g. Laguerre-Gauss, Speckles, Polychromatic). The package includes numerical tools to detect and trace polarization singularities in three dimensions of the generated fields.

## Requirements
- `numpy`
- `scipy`
- `numba` (Optional, but highly recommended for computational speed)
- `matplotlib` (For visualization)
- `tqdm` (For progress tracking)

## Library Workflow

The computation pipeline is strictly divided into three stages:
1. **Configuration**: Physical parameters (grid size, wavelength, spatial/spectral profiles, randomness) are defined using validated dataclasses.
2. **Setup**: The configuration is translated into a discrete set of plane waves (wavevectors, complex coefficient vector).
3. **Engine**: The resulting beam is routed to a computational backend (`numba` or `numpy`) to evaluate the fields on 2D grids, 3D point clouds, or single spatial points.

## Basic Usage

### 1. Simulating a simple Optical Field
This example generates Intensity of a Gaussian spatial profile and a Lorentzian spectral envelope.

```python
import numpy as np
import matplotlib.pyplot as plt
from wavelab0 import get_config, setup_experiment

# 1. Define configuration
config = get_config()
config.op.size = (5.0,5.0)         # 10x10 unit^2 Observation Plane
config.source.num_modes = 15000    # Number of plane waves

# Define spectral and spatial distributions
config.spectrum.spatial.gaussian(sigma_k_perp=1.5)
config.spectrum.spectral.lorentzian(center=0.5, gamma=0.1)

# Wavelengths around the lorentzian center
config.source.wavelengths = np.linspace(0.2,0.8,11) 

# Turn off stochastic effects
config.random.off()

# 2. Initialize setup and engine
expt = setup_experiment(config)

# 3. Compute fields on the observation plane
E, (dE_dx, dE_dy, dE_dz), B = expt.compute_on_op(z=0.0,t=0.0)

# Plot Intensity
I = np.sum(np.abs(E)**2, axis=0)
op_extent = expt.op_extent

plt.imshow(I, cmap='inferno',extent=op_extent)
plt.title('Field Intensity')
plt.show()
```

### 2. Finding C points in a Speckle Field
This example generates stokes paramters of a speckle field and finds C points.

```python
import numpy as np
import matplotlib.pyplot as plt
from wavelab0 import get_config, setup_experiment, get_stokes_parameters, SingularityFinder

# 1. Define configuration
config = get_config()
config.op.size = (5.0, 5.0)      # 10x10 unit^2 Observation Plane
config.source.num_modes = 15000  # Number of plane waves
config.source.wavelengths = 1    # Monochromatic

# Define spatial distributions
config.spectrum.spatial.gaussian(sigma_k_perp=1.5)

# Keep stochastic effects - default state of config creates speckles!

# 2. Initialize setup and engine
expt = setup_experiment(config)

# 3. Compute fields on the observation plane
E, _, _ = expt.compute_on_op(z=0.0,t=0.0,need_b=False,need_derivs=False)
s_all = get_stokes_parameters(E,normalize=True) # for s3

# 4. Find C points inside the observation plane
finder = SingularityFinder(expt)
c_points = finder.find_stokes_C_points(z_value=0,E_grid=E)

# Extract their positions and handedness from confident data.
cp_positions = [point['position'] for point in c_points if point['confident']]
cp_handedness = [point['handedness'] for point in c_points if point['confident']]

x, y, z = zip(*cp_positions)

# Plot s3 Field and C points.
op_extent = expt.op_extent
plt.imshow(s_all['s3'],cmap='coolwarm',origin='lower',extent=op_extent)
plt.scatter(x,y,c=cp_handedness)
plt.legend()
plt.title('s3 field with C points marked.')
plt.show()
```
