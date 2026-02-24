"""
Field Engine
============

Orchestrates the computation of electromagnetic fields by routing the 
precomputed Beam object to the chosen hardware backend.
"""
import time
import numpy as np

from .config import Config
from .setup import Beam

# Import backends
from .backends.numpy_backend import NumpyMethods
try:
    from .backends.numba_backend import NumbaMethods, has_numba
except ImportError:
    NumbaMethods = None
    has_numba = False


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
        self.X, self.Y = np.meshgrid(self.x,self.y)

        self.op_extent = [min(self.x),max(self.x),min(self.y),max(self.y)]

    def selector(self, choice: str):
        choice = choice.lower()
        if choice == "auto":
            return "numba" if has_numba else "numpy"
        
        elif choice == "numba":
            if not has_numba: 
                raise ValueError("Numba backend is requested, but it is unavailable")
            else: 
                return "numba"
            
        elif choice == "numpy": 
            return "numpy"
        
        else: 
            raise ValueError(f"Backend '{choice}' is not supported.")

    def get_backend(self, backend_name = None):
        if not backend_name: backend_name = self.backend_name
        if backend_name == "numba":
            return NumbaMethods(self.beam)
        elif backend_name == "numpy":
            return NumpyMethods(self.beam)
        else: 
            raise ValueError(f"Unknown backend: {backend_name}")
        
    def compute_on_op( 
            self, z: float = 0.0, t: float = 0.0, 
            need_b: bool = True, need_derivs: bool = True, 
            backend_name: str | None = None,
        ):
        """
        Computes the electromagnetic field arrays on the Observation Plane (OP).

        The observation plane dimensions and resolution are determined by the 
        `Config.op` settings passed during initialization.

        Parameters
        ----------
        z : The longitudinal z-coordinate of the observation plane. Default is 0.0.
        t : The time step for the field evaluation. Default is 0.0.
        need_b : If True, calculates and returns the Magnetic field (B). Default is True.
        need_derivs : If True, calculates and returns the spatial derivatives of E. Default is True.
        backend_name : Optional, for switching backend post initialization.

        Returns
        -------
        E : np.ndarray
            Electric field array of shape (3, Ny, Nx).
        D : tuple of np.ndarray or tuple of Nones
            Spatial derivatives (dE_dx, dE_dy, dE_dz), each of shape (3, Ny, Nx).
        B : np.ndarray or None
            Magnetic field array of shape (3, Ny, Nx).
        """

        backend = self.get_backend(backend_name)

        if self.config.verbose:
            print(
                f"OP Grid: {len(self.x)}x{len(self.y)} points | "
                f"Spacing: {self.config.op.spacing} | Z-plane: {z}"
            )
            t0 = time.time()

        E, D, B = backend.compute_grid(
            self.x, self.y, z, t,
            need_b=need_b, need_derivs=need_derivs,
            progress_bar=self.config.verbose,
        )

        if self.config.verbose:
            print(f"OP Computation complete in {time.time() - t0:.4f}s")

        return E, D, B    
    
    def compute_grid(
            self, x_vec: np.ndarray, y_vec: np.ndarray, z: float = 0.0, t: float = 0.0,
            need_b: bool = True, need_derivs: bool = True, backend_name: str | None = None,
        ):
        """Direct wrapper to compute on a custom 2D grid plane."""
        backend = self.get_backend(backend_name)
        
        return backend.compute_grid(
            x_vec, y_vec, z, t, 
            need_b=need_b, need_derivs=need_derivs, 
            progress_bar=self.config.verbose,
        )
    
    def compute_cloud(
            self, x_arr: np.ndarray, y_arr: np.ndarray, z_arr: np.ndarray, t: float = 0.0,
            need_b: bool = True, need_derivs: bool = True, backend_name: str | None = None,
        ):
        """       
        Field calculator for N arbitrary points (cloud).
        Does NOT support multidimensional grids (use flattening if needed).

        Parameters
        ----------
        x_arr,y_arr,z_arr : Coordinates specifying point to evaluate. Each can be 1D array at max.
        t: Time coordinate.
        need_derivs : If False, returns None in place of derivs.
        need_b: If False, returns None in place of B.
        backend_name: Optional, for switching backend post initialization.

        Returns
        -------
        E : np.ndarray
            Electric field array of shape (3, N).
        D : tuple of np.ndarray or tuple of Nones
            Spatial derivatives (dE_dx, dE_dy, dE_dz), each of shape (3, N).
        B : np.ndarray or None
            Magnetic field array of shape (3, N).
        """

        backend = self.get_backend(backend_name)

        return backend.compute_cloud(
            x_arr, y_arr, z_arr, t,
            need_b=need_b, need_derivs=need_derivs,
            progress_bar=self.config.verbose,
        )
    
    def compute_point(
            self, x: float, y: float, z: float, t: float = 0.0,
            need_b: bool = True, need_derivs: bool = True, backend_name: str | None = None,
        ):
        """
        Compute fields at a single exact spacetime point.

        Parameters
        ----------
            x,y,z: Coordinate specifying point to evaluate. 
            t: Time coordinate.
            need_derivs : If False, returns None in place of derivs.
            need_b: If False, returns None in place of B.
            backend_name: Optional, for switching backend post initialization.

        Returns
        -------
        E : np.ndarray
            Electric field array of shape (3).
        D : tuple of np.ndarray or tuple of Nones
            Spatial derivatives (dE_dx, dE_dy, dE_dz), each of shape (3).
        B : np.ndarray or None
            Magnetic field array of shape (3).

        """

        backend = self.get_backend(backend_name)

        return backend.compute_point(x, y, z, t, need_b=need_b, need_derivs=need_derivs)