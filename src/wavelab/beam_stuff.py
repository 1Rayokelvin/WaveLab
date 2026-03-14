"""
Beam Representation & Generation
================================

Translates experiment configurations into physical beam objects. This module 
contains the `Beam` dataclass, which stores precomputed coefficients and 
angular spectrum data, and the `BeamMaker` factory.

Pipeline Context:
    1. Config (config_stuff.py) -> Provided as input to BeamMaker.
    2. Beam (beam_stuff.py)     -> **CURRENT STEP**. Precomputes spectrum.
    3. Engine (engine_stuff.py) -> Resulting Beam is passed to FieldEngine.
"""

import numpy as np
import warnings
from dataclasses import dataclass
from typing import Tuple, Optional, Literal
from functools import cached_property

from .config_stuff import Config

@dataclass
class Beam:
    """
    Structured representation of a decomposed electromagnetic beam.
    
    This class serves two purposes:
    1. Storage: Holds the wavevectors (k) and complex 
       vector amplitudes (c) used by backends for field superposition.
    2. Physics Analysis: Provides properties and visualization tools to 
       inspect the beam's divergence, power, and spectrum.

    Attributes
    ----------
    k : np.ndarray
        (3, N) Wavevectors in units of rad/spatial_unit.
    c : np.ndarray
        (3, N) Complex vector amplitudes incorporating polarization, 
        intensity scaling, and phase offsets.
    w : np.ndarray
        (N,) Angular frequencies (norm of k).
    inv_w : np.ndarray
        (N,) Precomputed inverse frequencies for normalization.
    a : np.ndarray
        (N,) Scalar complex amplitudes (A * exp(i*phi)) before polarization.
    """
    k: np.ndarray       
    c: np.ndarray       
    w: np.ndarray       
    inv_w: np.ndarray   
    a: np.ndarray       

    # =========================================================================
    #                       PHYSICS PROPERTIES
    # =========================================================================

    @cached_property
    def num_modes(self) -> int:
        """Total number of plane wave modes in the beam."""
        return self.k.shape[1]

    @cached_property
    def wavelengths(self) -> np.ndarray:
        """Physical wavelengths of each mode (N,)."""
        wl = np.zeros_like(self.w)
        mask = self.w > 0
        wl[mask] = 2 * np.pi / self.w[mask]
        return wl

    @cached_property
    def k_hat(self) -> np.ndarray:
        """Normalized propagation direction unit vectors (3, N)."""
        return self.k * self.inv_w

    @cached_property
    def amplitudes(self) -> np.ndarray:
        """The real-valued magnitude of each mode (Scalar)."""
        return np.abs(self.a)

    @cached_property
    def polarizations(self) -> np.ndarray:
        """
        The Jones vectors (unit complex vectors) of each mode.
        Extracted by dividing the vector amplitude (c) by the scalar amplitude (a).
        """
        mask = np.abs(self.a) > 1e-15
        pol = np.zeros_like(self.c, dtype=complex)
        pol[:, mask] = self.c[:, mask] / self.a[mask]
        return pol

    @cached_property
    def mode_irradiances(self) -> np.ndarray:
        """
        The spatiotemporally averaged irradiance contribution of each mode.
        
        For a single plane wave, the time-averaged energy density is uniform 
        across all space. This value represents the 'weight' of each discrete 
        spectral component in the superposition.

        Calculated as the squared norm of the complex vector amplitudes: 
        |c_x|^2 + |c_y|^2 + |c_z|^2.
        """
        return np.sum(np.abs(self.c)**2, axis=0)

    @cached_property
    def total_power(self) -> float:
        """
        The integrated spectral norm of the beam.
        
        Calculated as the sum of all individual mode irradiances,
        it represents the total energy content of the angular spectrum. 
        
        While the local spatial intensity (FieldResult.intensity_E) is shaped by 
        interference, this value is a conserved quantity that defines the 'bulk' 
        strength of the beam. 
        """
        return float(np.sum(self.mode_irradiances)) 
       
    # =========================================================================
    #                       USER TOOLS & VISUALIZATION
    # =========================================================================

    def summary(self):
        """Prints a physical summary of the beam including divergence and axis."""
        print(f"--- Beam Physics Summary ---")
        print(f"Modes          : {self.num_modes:,}")
        print(f"Total Power    : {self.total_power:.2e}")
        
        wls = np.unique(np.round(self.wavelengths, 5))
        if len(wls) == 1:
            print(f"Wavelength     : {wls[0]} (Monochromatic)")
        elif len(wls) < 10:
            print(f"Wavelengths    : {wls} ({len(wls)} distinct lines)")
        else:
            print(f"Wavelengths    : {np.min(wls)} to {np.max(wls)} (Broadband)")
            
        if self.total_power > 1e-15:
            weights = self.mode_irradiances / self.total_power
            mean_k = np.sum(self.k_hat * weights[np.newaxis, :], axis=1)
            axis_norm = np.linalg.norm(mean_k)
            
            if axis_norm > 0:
                mean_axis = mean_k / axis_norm
                cos_thetas = np.clip(np.dot(mean_axis, self.k_hat), -1.0, 1.0)
                thetas = np.arccos(cos_thetas)
                rms_theta = np.sqrt(np.sum(weights * thetas**2))
                
                print(f"Mean Axis      : [{mean_axis[0]:.3f}, {mean_axis[1]:.3f}, {mean_axis[2]:.3f}]")
                print(f"RMS Divergence : ~{np.degrees(rms_theta):.2f} degrees half-angle")

    def plot_kspace_3d(
            self,  cmap='inferno', plot_type:Literal['colored_vectors','colored_sphere']='colored_vectors'
            ):
        """
        Renders an interactive 3D visualization of the wavevectors, amplitudes using PyVista..

        Parameters
        ----------
        cmap : optional
            Colormap for mode amplitudes (default 'inferno').
        plot_type : Literal['colored_vectors', 'colored_sphere'], optional
            'colored_vectors': arrows along k_hats colored by amplitude.
            'colored_sphere': unit sphere colored by amplitudes for a smooth heatmap.
            Default is colored_vectors.

        Returns
        -------
        pyvista.Plotter
            Plotter object for further manipulation.
        """
        try:
            import pyvista as pv
        except ImportError:
            warnings.warn("pyvista is required for 3D visualization.")
            return
        
        plotter = pv.Plotter()
        plotter.set_scale(1)
        plotter.show_axes()


        if plot_type == 'colored_vectors':
            origins = np.zeros((self.num_modes, 3))
            mesh = pv.PolyData(origins)
            mesh["vec"] = self.k_hat.T
            mesh["amplitudes"] = self.amplitudes
            arrows = mesh.glyph(orient="vec", scale=False, factor=0.2)
            plotter.add_mesh(
                arrows, scalars='amplitudes', 
                cmap=cmap, clim=[0, np.max(self.amplitudes)],
                scalar_bar_args={'vertical': True, 'title': 'Amplitude'}
                )

        elif plot_type == 'colored_sphere':
            sphere = pv.Sphere(theta_resolution=60, phi_resolution=120)
            # map amplitudes to sphere points using nearest-neighbor
            from scipy.spatial import cKDTree
            tree = cKDTree(self.k_hat.T)
            _, idx = tree.query(sphere.points)  # find nearest k_hat for each sphere point
            sphere["amplitudes"] = self.amplitudes[idx]
            plotter.add_mesh(
                sphere, scalars='amplitudes', 
                cmap=cmap, clim=[0, np.max(self.amplitudes)],
                scalar_bar_args={'vertical': True, 'title': 'Amplitude'}
                )
            
        else: raise ValueError("plot_type must be colored_sphere or colored_vectors")

        plotter.show()
        
        return plotter
    
    def plot_k_perp_profile(self, normal: Optional[Tuple[float, float, float]] = None, show: bool = True):
        """
        Plots Amplitude vs Transverse wave number (k_perp).

        Parameters
        ----------
        normal : tuple, optional
            The normal vector defining the longitudinal axis. If None, it attempts 
            to find the intensity-weighted mean direction. If the beam is 
            perfectly symmetric (e.g., a standing wave), it falls back to the 
            direction of the dominant mode.
        show: bool
            If true, displays the plot.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("matplotlib is required to plot k-space profiles.")
            return None
            
        if normal is not None:
            normal_vec = np.array(normal, dtype=float)
            norm = np.linalg.norm(normal_vec)
            normal_vec = normal_vec / norm if norm > 0 else np.array([0., 0., 1.])
        else:
            # 1. Calculate weighted mean direction
            weights = self.mode_irradiances / self.total_power
            mean_k = np.sum(self.k_hat * weights[np.newaxis, :], axis=1)
            mean_norm = np.linalg.norm(mean_k)
            
            # 2. Safety check for zero-mean
            if mean_norm < 1e-5:
                # Fallback: Use the axis of the mode with the highest irradiance
                max_idx = np.argmax(self.mode_irradiances)
                normal_vec = self.k_hat[:, max_idx]
            else:
                normal_vec = mean_k / mean_norm
            
        # 3. Compute k_perp relative to the identified normal
        # k_perp = |k - (k·n)n|
        k_parallel_mags = np.dot(normal_vec, self.k)
        k_parallel_vecs = normal_vec[:, np.newaxis] * k_parallel_mags
        k_perp = np.linalg.norm(self.k - k_parallel_vecs, axis=0)
        
        fig, ax = plt.subplots(figsize=(7, 4))
        
        ax.scatter(k_perp, self.amplitudes, s=15)
        ax.set_xlabel(r'Transverse Wavenumber $k_\perp$')
        ax.set_ylabel('Mode Amplitude')
        ax.set_title(f"K-Space Tranverse Profile about\nNormal: [{normal_vec[0]:.2f}, {normal_vec[1]:.2f}, {normal_vec[2]:.2f}]")
        ax.grid(True, alpha=0.2)
        ax.set_ybound(lower=0,upper=np.max(self.amplitudes)*1.2)
        plt.tight_layout()
        
        if show:
            plt.show()
        return fig, ax    
    
    def plot_wavelength_spectrum(self, show: bool = True):
        """Plots the intensity-weighted wavelength spectrum."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("matplotlib is required to plot spectrum.")
            return None
            
        fig, ax = plt.subplots(figsize=(6, 4))
        wls = np.round(self.wavelengths, decimals=5)
        unique_wls = np.unique(wls)
        
        if len(unique_wls) == 1:
            ax.axvline(unique_wls[0], color='indigo', lw=3, label=fr'$\lambda={unique_wls[0]}$')
        else:
            spectra = [np.sum(self.mode_irradiances[wls == wl]) for wl in unique_wls]
            ax.bar(unique_wls, spectra, width=max(np.ptp(unique_wls)*0.02, 1e-3), color='indigo')
            
        ax.set_xlabel("Wavelength")
        ax.set_ylabel("Intensity")
        if show: plt.show()
        return fig, ax

class BeamMaker:
    """
    Factory class that translates a `Config` object into a `Beam`.
    
    Handles the mathematical heavy lifting of sphere sampling (Fibonacci), 
    polarization basis construction (Rodrigues), and spectral weight application.
    """
    def __init__(self, config: Config):
        self.config = config
        self.config.validate()
        self.rng = np.random.default_rng(config.source.randomize.seed)
        
    def generate_beam(self) -> Beam:
        """
        Executes the generation pipeline to produce a superposition of plane waves.
        
        Returns
        -------
        Beam
            Precomputed beam object ready for evaluation in the FieldEngine.
        """
        if self.config.verbose:
            print(f"--- Starting Beam Generation (Modes: {self.config.source.num_modes}) ---")

        wls = np.atleast_1d(self.config.source.wavelength)
        num_wls = len(wls)

        # 1. Compute polychromatic envelope weights
        if num_wls > 1:
            poly_cfg = self.config.source.polychromatic
            weights = np.array([poly_cfg.profile(wl, **poly_cfg.params) for wl in wls])
            w_sum = np.linalg.norm(weights)
            weights = weights / w_sum if w_sum > 1e-12 else np.ones(num_wls) / np.sqrt(num_wls)
        else:
            weights = np.ones(num_wls)

        weights *= np.sqrt(self.config.source.intensity_scale)

        # 2. Generate sampling grid on unit sphere
        master_k_hats, master_d_omega = self._sample_sphere_fib(
            N=self.config.source.num_modes,
            beam_axis=self.config.source.beam_axis,
            theta_max=self.config.source.theta_max
        )

        # 3. Generate wave batches per wavelength
        all_ks, all_cs, all_amps = [], [], []

        for i, (wl, spectral_weight) in enumerate(zip(wls, weights)):
            indices = slice(i, None, num_wls)
            k_chunk = master_k_hats[indices]
            d_omega_chunk = master_d_omega[indices]
            if len(k_chunk) == 0: continue

            ks, cs, amps = self._generate_monochromatic_batch(wl, k_chunk, d_omega_chunk, spectral_weight)
            all_ks.append(ks); all_cs.append(cs); all_amps.append(amps)

        # 4. Final Aggregation
        k_out = np.vstack(all_ks).T      
        c_out = np.vstack(all_cs).T      
        a_out = np.concatenate(all_amps) 
        w_out = np.linalg.norm(k_out, axis=0)
        
        with np.errstate(divide='ignore'):
            inv_w_out = 1.0 / w_out
        inv_w_out[w_out == 0] = 0

        return Beam(k=k_out, c=c_out, w=w_out, inv_w=inv_w_out, a=a_out)

    def _generate_monochromatic_batch(self, wavelength: float, k_hats: np.ndarray, 
                                      d_omega: np.ndarray, weight: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Internal helper for generating modes at a specific wavelength line."""
        N = len(k_hats)
        ks = (2 * np.pi / wavelength) * k_hats

        # --- Polarization Basis (Local Transverse Frames) ---
        e1, e2 = self._transverse_basis_batch_rod(k_hats, self.config.source.beam_axis)
        px, py = self.config.source.pol_vect
        
        # Handle Randomized Polarization
        if self.config.source.randomize.pol_state:
            temp = self._sample_sphere_fib(N, (0, 0, 1), np.pi)
            s1, s2, s3 = temp[0][:, 0], temp[0][:, 1], temp[0][:, 2]
            P = np.sqrt((1.0+s1)/2.0)[:,None]*e1 + (np.sqrt((1.0-s1)/2.0)*np.exp(1j*np.arctan2(s3, s2)))[:,None]*e2
        elif self.config.source.randomize.pol_rot:
            angles = self.rng.uniform(0, 2*np.pi, size=N)
            c_a, s_a = np.cos(angles), np.sin(angles)
            P = (c_a*px - s_a*py)[:,None]*e1 + (s_a*px + c_a*py)[:,None]*e2
        else:
            P = px * e1 + py * e2

        # --- K-Space Amplitude Spectrum ---
        kspace_cfg = self.config.source.k_space
        if kspace_cfg.vectorised:
            amps = np.asarray(kspace_cfg.profile(ks.T, **kspace_cfg.params), dtype=complex).squeeze()
        else:
            amps = np.array([kspace_cfg.profile(k, **kspace_cfg.params) for k in ks], dtype=complex)

        # --- Stochastic Noise ---
        if self.config.source.randomize.phase:
            amps *= np.exp(1j * 2 * np.pi * self.rng.random(N))
        if self.config.source.randomize.amplitude:
            amps *= (self.rng.normal(0,1,N) + 1j*self.rng.normal(0,1,N)) * 0.7071

        # --- Power Normalization ---
        raw_power = np.sum(np.abs(amps)**2 * d_omega)
        if raw_power < 1e-15:
            return ks, np.zeros((N, 3), dtype=complex), np.zeros(N, dtype=complex)
            
        scaling = (1.0 / np.sqrt(raw_power)) * weight * d_omega 
        amps *= scaling
        return ks, P * amps[:, np.newaxis], amps

    # =========================================================================
    #                       SAMPLING STRATEGIES
    # =========================================================================    
    def _sample_sphere_fib(self, N: int, beam_axis: Tuple, theta_max: float) -> Tuple[np.ndarray, np.ndarray]:
        z_min = np.cos(theta_max)
        z_range = 1.0 - z_min
        i = np.arange(N)
        z = 1.0 - (i + 0.5) * z_range / N
        r = np.sqrt(np.maximum(0, 1 - z**2))
        phi = np.pi * (3.0 - np.sqrt(5.0)) * i
        
        points = np.column_stack((r * np.cos(phi), r * np.sin(phi), z))
        ang = self.rng.uniform(0, 2*np.pi)
        c, s = np.cos(ang), np.sin(ang)
        points = points @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T

        return self._align_to_axis(points, beam_axis), np.full(N, (2 * np.pi * z_range) / N)

    def _align_to_axis(self, points: np.ndarray, target_axis: Tuple) -> np.ndarray:
        target = np.array(target_axis)
        norm = np.linalg.norm(target)
        if norm == 0: return points
        target = target / norm
        z_hat = np.array([0.0, 0.0, 1.0])
        c = np.dot(z_hat, target)
        if c > 0.999999: return points
        if c < -0.999999:
            p2 = points.copy()
            p2[:, 2] *= -1; p2[:, 0] *= -1 
            return p2
        v = np.cross(z_hat, target)
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + (vx @ vx) * ((1 - c) / np.dot(v, v))
        return points @ R.T

    def _transverse_basis_batch_rod(self, ks: np.ndarray, beam_axis: Tuple) -> Tuple[np.ndarray, np.ndarray]:
        beam_axis = np.array(beam_axis)
        n = beam_axis / np.linalg.norm(beam_axis) if np.linalg.norm(beam_axis) > 0 else np.array([0., 0., 1.])
        ks_norm = ks / np.linalg.norm(ks, axis=1, keepdims=True)
        
        u = np.cross(n, [0.0, 0.0, 1.0] if np.abs(n[2]) < 0.9 else [0.0, 1.0, 0.0])
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        
        w = np.cross(n, ks_norm)
        s = np.linalg.norm(w, axis=1, keepdims=True)
        c = np.sum(n * ks_norm, axis=1, keepdims=True)
        
        e1, e2 = np.tile(u, (len(ks), 1)), np.tile(v, (len(ks), 1))
        mask = s[:, 0] > 1e-9
        if np.any(mask):
            wn = w[mask] / s[mask]
            u_dot, v_dot = np.sum(wn * u, axis=1, keepdims=True), np.sum(wn * v, axis=1, keepdims=True)
            e1[mask] = (u * c[mask] + np.cross(wn, u) * s[mask] + wn * u_dot * (1 - c[mask]))
            e2[mask] = (v * c[mask] + np.cross(wn, v) * s[mask] + wn * v_dot * (1 - c[mask]))
            
        return e1, e2
