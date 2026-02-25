"""
setup.py
================

Make the beam described by a config data class made in config.py
"""
import numpy as np
from dataclasses import dataclass
from typing import Tuple
import time
from .config import Config

@dataclass
class Beam:
    """
    Precomputed physical properties of the generated electromagnetic field.
    
    Attributes
    ----------
    k : np.ndarray
        Wavevectors of shape (3, N).
    c : np.ndarray
        Precomputed complex phasor vectors (Polarization * Amplitude) of shape (3, N).
    w : np.ndarray
        Angular frequencies (c * |k|, with c=1) of shape (N,).
    inv_w : np.ndarray
        Inverse frequencies (1 / w) of shape (N,).
    """
    k: np.ndarray
    c: np.ndarray
    w: np.ndarray
    inv_w: np.ndarray

class ExperimentSetup:
    """
    Stateless builder class that translates a physical Config into a computed Beam.
    """
    def __init__(self, config: Config):
        self.config = config
        self.config.validate()
        self.rng = np.random.default_rng(config.random.seed)
        
    def generate_beam(self) -> Beam:
        """
        Main pipeline to generate the superposition of plane waves.
        Handles polychromatic weighting and aggregates all modes into a single Beam object.
        """
        if self.config.verbose:
            print(f"--- Starting Beam Generation (Modes: {self.config.source.num_modes}) ---")

        wls = np.atleast_1d(self.config.source.wavelength)
        num_wls = len(wls)

        # 1. Compute polychromatic envelope weights
        if num_wls > 1 and self.config.spectrum.spectral.profile:
            weights = np.array([
                self.config.spectrum.spectral.profile(wl, **self.config.spectrum.spectral.params)
                for wl in wls
            ], dtype=float)
            
            if np.sum(weights) < 1e-12:
                raise ValueError("Spectral envelope is near-zero for all wavelengths.")
        else:
            weights = np.ones(num_wls, dtype=float)

        # Normalize weights and scale by total intensity
        weights = weights / np.linalg.norm(weights)
        weights *= np.sqrt(self.config.source.intensity_scale)

        # 2. Sample global angular support (Fibonacci sphere)
        master_k_hats = self._sample_sphere_fib(
            self.config.source.num_modes, 
            self.config.source.beam_axis
        )

        # 3. Generate wave batches per wavelength
        all_ks = []
        all_cs = []

        for i, (wl, weight) in enumerate(zip(wls, weights)):
            # Interleave k-vectors to ensure uniform angular distribution per wavelength
            k_chunk = master_k_hats[i::num_wls]
            
            if len(k_chunk) == 0:
                continue

            ks, cs = self._generate_monochromatic_batch(wl, k_chunk, weight)
            all_ks.append(ks)
            all_cs.append(cs)

            if self.config.verbose:
                print(f"  > Wavelength: {wl:.2f} | Modes: {len(k_chunk)} | Weight: {weight:.3e}")

        # 4. Aggregate and construct final Beam dataclass
        ks_stacked = np.vstack(all_ks)  # Shape: (N, 3)
        cs_stacked = np.vstack(all_cs)  # Shape: (N, 3)

        # Transpose to backend-friendly shapes: (3, N)
        k_out = ks_stacked.T
        c_out = cs_stacked.T
        
        # Calculate frequencies
        w_out = np.linalg.norm(k_out, axis=0)
        inv_w_out = 1.0 / w_out

        return Beam(k=k_out, c=c_out, w=w_out, inv_w=inv_w_out)

    def _generate_monochromatic_batch(self, wavelength: float, k_hats: np.ndarray, weight: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates k-vectors and complex c-vectors for a specific wavelength.
        """
        N = len(k_hats)
        k_mag = 2 * np.pi / wavelength

        # --- Geometry / Angular Support ---
        if self.config.source.angular_support == 'hemisphere':
            beam_axis = np.array(self.config.source.beam_axis)
            mask = np.einsum("ij,j->i", k_hats, beam_axis) < 0
            k_hats[mask] *= -1

        ks = k_mag * k_hats

        # --- Polarization Vectors (P) ---
        e1, e2 = self._transverse_basis_batch_rod(k_hats, self.config.source.beam_axis)
        px, py = self.config.source.pol_vect
        
        if self.config.random.pol_state:
            # Poincare Sphere Randomization
            temp = self._sample_sphere_fib(N, None, randomize=True)
            s1, s2, s3 = temp[:, 0], temp[:, 1], temp[:, 2]
            
            amp_x = np.sqrt((1.0 + s1) / 2.0)
            amp_y = np.sqrt((1.0 - s1) / 2.0)
            phase_diff = np.arctan2(s3, s2)
            
            pxs = amp_x.astype(complex)
            pys = amp_y * np.exp(1j * phase_diff)
            P = pxs[:, None] * e1 + pys[:, None] * e2

        elif self.config.random.pol_rot:
            # Random uniform rotation
            angles = self.rng.uniform(0, 2 * np.pi, size=N)
            c_a, s_a = np.cos(angles), np.sin(angles)
            pxr = c_a * px - s_a * py
            pyr = s_a * px + c_a * py
            P = pxr[:, None] * e1 + pyr[:, None] * e2

        else:
            # Fixed polarization
            P = px * e1 + py * e2

        # --- Spatial Amplitude Spectrum ---
        spatial_cfg = self.config.spectrum.spatial
        
        if spatial_cfg.vectorised:
            # Pass (3, N) array to vectorized spatial functions
            amps = spatial_cfg.profile(ks.T, **spatial_cfg.params)
            amps = np.asarray(amps, dtype=complex).squeeze()
        else:
            # Python loop for unvectorized custom functions
            amps = np.array([
                spatial_cfg.profile(k_vec, **spatial_cfg.params) 
                for k_vec in ks
            ], dtype=complex)

        # --- Stochastic Amplitude & Phase ---
        if self.config.random.phase:
            amps *= np.exp(1j * 2 * np.pi * self.rng.random(N))
            
        if self.config.random.complex_amplitude:
            amps *= (self.rng.normal(0, 1, N) + 1j * self.rng.normal(0, 1, N)) * (1.0 / np.sqrt(2))

        # --- Normalization & Scaling ---
        amp_norm = np.linalg.norm(amps)
        if amp_norm > 1e-12:
            amps = amps / amp_norm
        else:
            raise ValueError("Spatial spectrum is near-zero for sampled k-points.")

        # Riemann sum factor: Solid angle d_omega
        d_omega = 4.0 * np.pi / N
        amps *= weight * np.sqrt(d_omega)

        # Calculate final c-vector (Polarization * Amplitude)
        # P is (N, 3), amps is (N,), result is (N, 3)
        c_vectors = P * amps[:, np.newaxis]

        return ks, c_vectors

    def _sample_sphere_fib(self, N: int, beam_axis: Tuple[float, float, float] = None, randomize: bool = True) -> np.ndarray:
        """
        Generates uniform directions on the unit sphere using Fibonacci sampling.
        """
        i = np.arange(N)
        phi = np.pi * (3.0 - np.sqrt(5))
        theta = phi * i
        z = 1 - (2 * i + 1) / N
        r = np.sqrt(1 - z**2)
        
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points = np.column_stack((x, y, z))
        
        # Apply a global random seed rotation
        if randomize:
            angle = self.rng.uniform(0, 2 * np.pi)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            x_rot = cos_a * x + sin_a * y
            y_rot = -sin_a * x + cos_a * y
            points = np.column_stack((x_rot, y_rot, z))

        # Align poles to beam axis using Rodrigues rotation
        if beam_axis is not None:
            beam_axis = np.array(beam_axis)
            beam_axis = beam_axis / np.linalg.norm(beam_axis)
            
            z_hat = np.array([0.0, 0.0, 1.0])
            if np.abs(np.dot(beam_axis, z_hat)) < 0.9:
                new_pole = np.cross(beam_axis, z_hat)
            else:
                new_pole = np.cross(beam_axis, np.array([1.0, 0.0, 0.0]))
            
            new_pole = new_pole / np.linalg.norm(new_pole)
            v = np.cross(z_hat, new_pole)
            s = np.linalg.norm(v)
            c = np.dot(z_hat, new_pole)
            
            if s > 1e-10:
                vx = np.array([
                    [0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]
                ])
                R = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s ** 2))
                points = points @ R.T
        
        return points

    def _transverse_basis_batch_rod(self, K: np.ndarray, beam_axis: Tuple[float, float, float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the transverse polarization basis vectors for a batch of wavevectors.
        """
        beam_axis = np.array(beam_axis)
        K_norm = K / np.linalg.norm(K, axis=1, keepdims=True)
        n = beam_axis / np.linalg.norm(beam_axis)
        
        # Determine an arbitrary orthogonal vector 'u' to 'n'
        if np.abs(n[2]) < 0.9:
            u = np.cross(n, [0.0, 0.0, 1.0])
        else:
            u = np.cross(n, [0.0, 1.0, 0.0])
            
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        
        # Cross product of 'n' with wavevectors
        w = np.cross(n, K_norm)
        s = np.linalg.norm(w, axis=1, keepdims=True)
        c = np.sum(n * K_norm, axis=1, keepdims=True)
        
        e1 = np.copy(u * np.ones((len(K), 1)))
        e2 = np.copy(v * np.ones((len(K), 1)))
        
        # Rotate u, v to be orthogonal to local K using Rodrigues'
        mask = s[:, 0] > 1e-9
        if np.any(mask):
            wn = w[mask] / s[mask]
            
            e1[mask] = (u * c[mask] + 
                        np.cross(wn, u) * s[mask] + 
                        wn * np.sum(wn * u, axis=1, keepdims=True) * (1 - c[mask]))
            
            e2[mask] = (v * c[mask] + 
                        np.cross(wn, v) * s[mask] + 
                        wn * np.sum(wn * v, axis=1, keepdims=True) * (1 - c[mask]))
            
        return e1, e2