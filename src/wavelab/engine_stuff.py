"""
Field Engine & Results
======================

The final stage of the WaveLab pipeline. This module provides the `FieldEngine`, 
which orchestrates the spatial evaluation of electromagnetic fields by routing 
precomputed `Beam` objects to hardware backends (NumPy or Numba for now).

All computations are encapsulated in the `FieldResult` container, which provides 
a unified interface for Electric field components, vector calculus operations (Curl, 
Divergence), and the Jacobian tensor as well as the Magnetic field components.

Pipeline Context:
    1. Config (config_stuff.py) -> Define parameters.
    2. Beam (beam_stuff.py)     -> Precompute spectrum/weights via BeamMaker.
    3. Engine (engine_stuff.py) -> Evaluate fields on points/grids via FieldEngine.
"""

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config_stuff import Config
from .beam_stuff import Beam

# Import backends
from .backends.numpy_backend import NumpyMethods
try:
    from .backends.numba_backend import NumbaMethods, has_numba
except ImportError:
    NumbaMethods = None
    has_numba = False


@dataclass
class FieldResult:
    """
    Structured container for electromagnetic field data and its derivatives.

    This class provides a unified interface for accessing field components, 
    vector calculus operations (Divergence, Curl), and the Jacobian tensor 
    regardless of whether the evaluation was at a point, a cloud, or a grid.

    Shape Conventions
    -----------------
    Let `domain_shape` be the spatial dimensions of the evaluation:
    - compute_point: ()
    - compute_cloud: (N,)
    - compute_grid:  (Ny, Nx)

    - E, B:       (3, *domain_shape)
    - jacobian_E: (3, 3, *domain_shape)
    - div_E:      (*domain_shape)
    - curl_E:     (3, *domain_shape)

    Attributes
    ----------
    E : np.ndarray
        The Electric field vector. 
        E[0, ...] is Ex, E[1, ...] is Ey, E[2, ...] is Ez.
    """
    E: np.ndarray
    _B: Optional[np.ndarray] = None
    _jacobian_E: Optional[np.ndarray] = None

    def __repr__(self) -> str:
        shape = self.E.shape[1:] if self.E.ndim > 1 else "Point"
        has_b = self._B is not None
        has_jac = self._jacobian_E is not None
        return f"<FieldResult: Grid={shape}, B_computed={has_b}, Derivs_computed={has_jac}>"

    @property
    def B(self) -> np.ndarray:
        """
        The Magnetic field vector (3, *domain_shape).
        Raises RuntimeError if 'need_b' was False during computation.
        """
        if self._B is None:
            raise RuntimeError("Magnetic field (B) was not computed. Set need_b=True.")
        return self._B

    @property
    def jacobian_E(self) -> np.ndarray:
        """
        The Jacobian matrix of the Electric field (3, 3, *domain_shape).
        Raises RuntimeError if 'need_derivs' was False during computation.
        
        Layout (Numerator Convention):
            The first index (i) corresponds to the E-field component (Ex, Ey, Ez).
            The second index (j) corresponds to the spatial derivative(d/dx, d/dy, d/dz).
            
            result[i, j, ...] = dE_i / dx_j
            
        Example:
            >>> # Get dEy / dz
            >>> dEy_dz = field.jacobian_E[1, 2]
        """
        if self._jacobian_E is None:
            raise RuntimeError("Derivatives were not computed. Set 'need_derivs=True'.")
        return self._jacobian_E

    # --- Convenience Slice Accessors ---
    @property
    def dE_dx(self) -> np.ndarray: return self.jacobian_E[:, 0, ...]
    
    @property
    def dE_dy(self) -> np.ndarray: return self.jacobian_E[:, 1, ...]
    
    @property
    def dE_dz(self) -> np.ndarray: return self.jacobian_E[:, 2, ...]

    # --- Derived Physical Quantities ---
    @property
    def div_E(self) -> np.ndarray:
        """
        Divergence: div(E) 
        Computed as the trace of the Electric field Jacobian.
        """
        return np.trace(self.jacobian_E, axis1=0, axis2=1)

    @property
    def curl_E(self) -> np.ndarray:
        """
        Curl: curl(E)
        Computed from the anti-symmetric components of the Electric field Jacobian.
        """
        j = self.jacobian_E
        return np.stack([
            j[2, 1, ...] - j[1, 2, ...],
            j[0, 2, ...] - j[2, 0, ...],
            j[1, 0, ...] - j[0, 1, ...]
        ], axis=0)

    @property
    def intensity_E(self) -> np.ndarray:
        """Calculates intensity = |E|^2 over domain."""
        return np.sum(np.abs(self.E)**2, axis=0)
    

class FieldEngine:
    """ 
    Main engine for computing spatial electromagnetic fields from a Beam.
    """
    def __init__(self, beam: Beam, config: Config):
        self.beam = beam
        self.config = config
        
        self.backend_name = self.selector(self.config.backend)
        if self.config.verbose:
            print(f"--- FieldEngine Initialized (Backend: {self.backend_name}) ---")

        center_x, center_y = self.config.op.center
        w, h = self.config.op.size
        dx = self.config.op.spacing
        
        nx = int(w / dx)
        ny = int(h / dx)
        
        self.x = np.linspace(center_x - w/2, center_x + w/2, nx)
        self.y = np.linspace(center_y - h/2, center_y + h/2, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        self.op_extent = [min(self.x), max(self.x), min(self.y), max(self.y)]

    def selector(self, choice: str) -> str:
        choice = choice.lower()
        if choice == "auto":
            return "numba" if has_numba else "numpy"
        elif choice == "numba":
            if not has_numba: 
                raise ValueError("Numba backend is requested, but it is unavailable")
            return "numba"
        elif choice == "numpy": 
            return "numpy"
        else: 
            raise ValueError(f"Backend '{choice}' is not supported.")

    def get_backend(self, backend_name: Optional[str] = None):
        if not backend_name: 
            backend_name = self.backend_name
        if backend_name == "numba":
            return NumbaMethods(self.beam)
        elif backend_name == "numpy":
            return NumpyMethods(self.beam)
        else: 
            raise ValueError(f"Unknown backend: {backend_name}")
        
    def _wrap_results(self, E: np.ndarray, D: Optional[tuple], B: Optional[np.ndarray]) -> FieldResult:
        """Helper to package raw backend output into FieldResult."""
        E.flags.writeable = False
        if B is not None: B.flags.writeable = False
        
        jacobian_E = None
        if D is not None:
            jacobian_E = np.stack(D, axis=1)
            jacobian_E.flags.writeable = False
        
        return FieldResult(E=E, _B=B, _jacobian_E=jacobian_E)
    
    def compute_on_op( 
            self, z: float = 0.0, t: float = 0.0, 
            need_b: bool = True, need_derivs: bool = True, 
            backend_name: Optional[str] = None,
        ) -> FieldResult:
        """
        Computes the electromagnetic field arrays on the Observation Plane (OP).

        The observation plane dimensions and resolution are determined by the 
        `Config.op` settings passed during initialization.

        Parameters
        ----------
        z : float
            The longitudinal z-coordinate of the observation plane.
        t : float
            The time step for the field evaluation.
        need_b : bool
            If True, calculates and returns the Magnetic field (B).
        need_derivs : bool
            If True, calculates and returns spatial derivatives (Jacobian).
        backend_name : str, optional
            Override the default backend for this specific call.

        Returns
        -------
        FieldResult
            Object containing E (3, Ny, Nx) and optionally B and Jacobian.
        """
        backend = self.get_backend(backend_name)

        if self.config.verbose:
            print(f"OP Grid: {len(self.x)}x{len(self.y)} points | Spacing: {self.config.op.spacing} | Z: {z}")
            t0 = time.time()

        E, D, B = backend.compute_grid(
            self.x, self.y, z, t,
            need_b=need_b, need_derivs=need_derivs,
            progress_bar=self.config.verbose,
        )

        if self.config.verbose:
            print(f"OP Computation complete in {time.time() - t0:.4f}s")

        return self._wrap_results(E, D, B)    
    
    def compute_grid(
            self, x_vec: np.ndarray, y_vec: np.ndarray, z: float = 0.0, t: float = 0.0,
            need_b: bool = True, need_derivs: bool = True, backend_name: Optional[str] = None,
        ) -> FieldResult:
        """
        Direct wrapper to compute on a custom 2D grid plane.
        
        Returns
        -------
        FieldResult
            Object containing E (3, len(y_vec), len(x_vec)) and optional fields.
        """
        backend = self.get_backend(backend_name)
        
        return self._wrap_results(backend.compute_grid(
            x_vec, y_vec, z, t, 
            need_b=need_b, need_derivs=need_derivs, 
            progress_bar=self.config.verbose,
        ))
    
    def compute_cloud(
            self, x_arr: np.ndarray, y_arr: np.ndarray, z_arr: np.ndarray, t: float = 0.0,
            need_b: bool = True, need_derivs: bool = True, backend_name: Optional[str] = None,
        ) -> FieldResult:
        """       
        Field calculator for N arbitrary points (cloud).

        Parameters
        ----------
        x_arr, y_arr, z_arr : np.ndarray
            1D arrays of shape (N,) specifying evaluation points.
        t : float
            Time coordinate.
        need_b : bool
            If True, includes Magnetic field.
        need_derivs : bool
            If True, includes spatial derivatives.

        Returns
        -------
        FieldResult
            Object containing E (3, N) and optional fields.
        """
        backend = self.get_backend(backend_name)

        return self._wrap_results(backend.compute_cloud(
            x_arr, y_arr, z_arr, t,
            need_b=need_b, need_derivs=need_derivs,
            progress_bar=self.config.verbose,
        ))
    
    def compute_point(
            self, x: float, y: float, z: float, t: float = 0.0,
            need_b: bool = True, need_derivs: bool = True, backend_name: Optional[str] = None,
        ) -> FieldResult:
        """
        Compute fields at a single exact spacetime point.

        Returns
        -------
        FieldResult
            Object containing E (3,) and optional fields.
        """
        backend = self.get_backend(backend_name)

        return self._wrap_results(backend.compute_point(
                x, y, z, t, need_b=need_b, 
                need_derivs=need_derivs
            ))