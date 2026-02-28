"""
beam_maker.py
================

Make the beam described by a config data class.
"""
import numpy as np
import warnings
from dataclasses import dataclass
from typing import Tuple
from .config import Config

@dataclass
class Beam:
    """
    Precomputed physical properties of the generated electromagnetic field.
    """
    k: np.ndarray       # (3, N) Wavevectors
    c: np.ndarray       # (3, N) Complex amplitudes (Pol * Amp * exp(i*Phase) )
    w: np.ndarray       # (N,) Angular frequencies
    inv_w: np.ndarray   # (N,) Inverse frequencies

class BeamMaker:
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
        """
        if self.config.verbose:
            print(f"--- Starting Beam Generation (Modes: {self.config.source.num_modes}) ---")

        wls = np.atleast_1d(self.config.source.wavelength)
        num_wls = len(wls)

        # 1. Compute polychromatic envelope weights
        if num_wls > 1:
            weights = np.array([
                self.config.spectrum.spectral.profile(wl, **self.config.spectrum.spectral.params)
                for wl in wls
            ], dtype=float)
            
            w_sum = np.linalg.norm(weights)
            if w_sum > 1e-12:
                weights = weights / w_sum
            else:
                weights = np.ones(num_wls) / np.sqrt(num_wls)
        else:
            weights = np.ones(num_wls, dtype=float)

        # Scale by total requested intensity
        weights *= np.sqrt(self.config.source.intensity_scale)

        # 2. GENERATE SAMPLING GRID        
        master_k_hats, master_d_omega = self._get_sampling_vectors(
            N=self.config.source.num_modes,
            method=self.config.source.sampling_method,
            axis=self.config.source.beam_axis,
            param=self.config.source.sampling_param
        )

        # 3. Generate wave batches per wavelength
        all_ks = []
        all_cs = []

        for i, (wl, spectral_weight) in enumerate(zip(wls, weights)):
            # Interleave indices to distribute angular modes across wavelengths
            indices = slice(i, None, num_wls)
            
            k_chunk = master_k_hats[indices]
            d_omega_chunk = master_d_omega[indices]
            
            if len(k_chunk) == 0:
                continue

            ks, cs = self._generate_monochromatic_batch(
                wl, k_chunk, d_omega_chunk, spectral_weight
            )
            all_ks.append(ks)
            all_cs.append(cs)

            if self.config.verbose:
                print(f"  > Wavelength: {wl:.2e} | Modes: {len(k_chunk)} | SpecWeight: {spectral_weight:.2e}")

        # 4. Aggregate
        if not all_ks:
            raise ValueError("No modes generated. Check number of modes vs wavelengths.")

        ks_stacked = np.vstack(all_ks)  # (N_total, 3)
        cs_stacked = np.vstack(all_cs)  # (N_total, 3)

        # Transpose to (3, N) for backend efficiency
        k_out = ks_stacked.T
        c_out = cs_stacked.T
        
        # Calculate frequencies (c=1 units)
        w_out = np.linalg.norm(k_out, axis=0)
        
        # Avoid division by zero
        with np.errstate(divide='ignore'):
            inv_w_out = 1.0 / w_out
        inv_w_out[w_out == 0] = 0

        return Beam(k=k_out, c=c_out, w=w_out, inv_w=inv_w_out)

    def _generate_monochromatic_batch(self, wavelength: float, k_hats: np.ndarray, 
                                      d_omega: np.ndarray, weight: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates k-vectors and complex c-vectors for a specific wavelength chunk.
        """
        N = len(k_hats)
        k_mag = 2 * np.pi / wavelength
        ks = k_mag * k_hats

        # --- Polarization Vectors (P) ---
        e1, e2 = self._transverse_basis_batch_rod(k_hats, self.config.source.beam_axis)
        px, py = self.config.source.pol_vect
        
        if self.config.random.pol_state:
            # Randomize Poincare sphere state
            temp = self._sample_sphere_fib(N, (0, 0, 1), np.pi)
            s1, s2, s3 = temp[:, 0], temp[:, 1], temp[:, 2]
            
            amp_x = np.sqrt((1.0 + s1) / 2.0).astype(complex)
            amp_y = np.sqrt((1.0 - s1) / 2.0).astype(complex)
            phase_diff = np.arctan2(s3, s2)
            
            # Broadcast shapes: (N, 1) * (N, 3)
            P = amp_x[:, None] * e1 + (amp_y * np.exp(1j * phase_diff))[:, None] * e2

        elif self.config.random.pol_rot:
            # Random rotation angle
            angles = self.rng.uniform(0, 2 * np.pi, size=N)
            c_a, s_a = np.cos(angles), np.sin(angles)
            
            px_rot = (c_a * px - s_a * py)
            py_rot = (s_a * px + c_a * py)
            P = px_rot[:, None] * e1 + py_rot[:, None] * e2

        else:
            # Fixed polarization
            # px, py are scalars here. Result (N, 3)
            P = px * e1 + py * e2

        # --- Spatial Amplitude Spectrum ---
        spatial_cfg = self.config.spectrum.spatial
        if spatial_cfg.vectorised:
            # (3, N) input
            amps = spatial_cfg.profile(ks.T, **spatial_cfg.params)
            amps = np.asarray(amps, dtype=complex).squeeze()
        else:
            amps = np.array([
                spatial_cfg.profile(k_vec, **spatial_cfg.params) 
                for k_vec in ks
            ], dtype=complex)

        # --- Stochastic Phase/Amplitude ---
        if self.config.random.phase:
            amps *= np.exp(1j * 2 * np.pi * self.rng.random(N))
            
        if self.config.random.complex_amplitude:
            # Rayleigh distribution for amplitude, Uniform for phase
            # Standard complex normal (CN(0, 1))
            noise = (self.rng.normal(0, 1, N) + 1j * self.rng.normal(0, 1, N)) * (1.0 / np.sqrt(2))
            amps *= noise

        # --- Normalization ---
        raw_power = np.sum(np.abs(amps)**2 * d_omega)
        
        if raw_power < 1e-15:
            warnings.warn("Beam profile has near-zero integrated power.", UserWarning)
            c_vectors = np.zeros((N, 3), dtype=complex)
        else:
            # Normalize so the integral = 1
            norm_factor = 1.0 / np.sqrt(raw_power)
            
            # Combine Normalization, Spectral Weight, and integration weight
            scaling = norm_factor * weight * np.sqrt(d_omega)
            
            # Combine Polarization and Amplitude
            c_vectors = P * (amps * scaling)[:, np.newaxis]

        return ks, c_vectors

    # =========================================================================
    #                       SAMPLING STRATEGIES
    # =========================================================================

    def _get_sampling_vectors(
            self, N: int, method: str, axis: Tuple, param: float
            ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Selects sampling strategy.
        
        Args:
            N: Number of modes
            method: 'gaussian' or 'fibonacci'
            axis: Beam central axis (x, y, z)
            param: For 'gaussian', this is sigma (spread). 
                     For 'fibonacci' it is theta_max (cutoff).
            
        Returns:
            points: (N, 3) unit vectors
            d_omega: (N,) integration weights
        """
        if method == 'gaussian':
            return self._sample_spherical_gaussian(N, axis, param)
        elif method == 'fibonacci':
            return self._sample_sphere_fib(N, axis, param)
        else: raise ValueError("Sampling mode must be 'gaussian' or 'fibonacci'")
            
    def _sample_sphere_fib(
            self, N: int, beam_axis: Tuple, theta_max: float
            ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Oversampled Fibonacci Sphere.
        
        Generates points on the full unit sphere, applies a random rotation (so the 
        Fibonacci poles don't align with the beam axis), and masks out points 
        outside the requested theta_max.
        """
        # 1. Calculate Area Fraction (Cap Area / Sphere Area)
        fraction = (1.0 - np.cos(theta_max)) / 2.0
        
        # If aperture is tiny (e.g. < 5 degrees), oversampling becomes expensive
        # Fallback to direct cap mapping in this physical limit.
        if N / (fraction + 1e-9) > 1e5:
            return self._sample_fib_cap_direct(N, beam_axis, theta_max)

        # 2. Oversample
        N_total = int(np.ceil(N / fraction * 1.2))
        
        # 3. Generate Full Sphere Fibonacci (Poles at +/- Z)
        i = np.arange(N_total)
        phi = np.pi * (3.0 - np.sqrt(5.0)) * i
        z = 1 - (2 * i + 1) / N_total
        r = np.sqrt(np.maximum(0, 1 - z**2))
        points = np.column_stack((r * np.cos(phi), r * np.sin(phi), z))
        
        # 4. Random Rotation
        ang = self.rng.uniform(0, 2*np.pi)
        c, s = np.cos(ang), np.sin(ang)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        points = points @ Rz.T

        # 5. Masking
        target = np.array(beam_axis)
        target /= np.linalg.norm(target)
        
        # Dot product > cos(theta_max) means inside the cone
        cos_sim = np.dot(points, target)
        mask = cos_sim >= np.cos(theta_max)
        valid_points = points[mask]
        
        # 6. Strict N Enforcement
        if len(valid_points) < N:
            # Should be rare given the 1.2x buffer. Recursive retry if it happens.
            return self._sample_sphere_fib(N, beam_axis, theta_max)
            
        # Shuffle to avoid bias from the index order (latitude bias)
        self.rng.shuffle(valid_points)
        final_points = valid_points[:N]
        
        # 7. Weights
        # The density of the grid was uniform over 4pi.
        # d_omega = Area_Sphere / N_total_generated
        d_omega = np.full(N, 4.0 * np.pi / N_total)
        
        return final_points, d_omega

    def _sample_fib_cap_direct(
            self, N: int, beam_axis: Tuple, theta_max: float
            ) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback: Direct mapping for very small apertures."""
        z_min = np.cos(theta_max)
        z_range = 1.0 - z_min
        
        i = np.arange(N)
        z = 1.0 - (i + 0.5) * z_range / N
        r = np.sqrt(np.maximum(0, 1 - z**2))
        phi = np.pi * (3.0 - np.sqrt(5.0)) * i
        
        points = np.column_stack((r * np.cos(phi), r * np.sin(phi), z))
        
        # Rotate randomly around local Z
        ang = self.rng.uniform(0, 2*np.pi)
        c, s = np.cos(ang), np.sin(ang)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        points = points @ Rz.T

        # Area of cap / N
        d_omega = np.full(N, (2 * np.pi * z_range) / N)
        return self._align_to_axis(points, beam_axis), d_omega
    
    def _sample_spherical_gaussian(
            self, N: int, beam_axis: Tuple, sigma: float
            ) -> Tuple[np.ndarray, np.ndarray]:
        
        # 1. Setup the Probability Distribution P(theta) = p(theta) * jacobian
        limit = min(np.pi, 5.0 * sigma)
        t_grid = np.linspace(0, limit, 1000)
        
        pdf_shape = np.exp(-0.5 * (t_grid / sigma)**2) * np.sin(t_grid)
        
        # 2. Compute Normalization Constant (Z)
        # This is the "Total Effective Area" of the sampling cone
        Z = np.trapz(pdf_shape, t_grid)
        
        # 3. Inverse Transform Sampling (CDF) to pick Theta
        cdf = np.cumsum(pdf_shape)
        cdf /= cdf[-1]
        
        u = self.rng.random(N)
        theta = np.interp(u, cdf, t_grid)
        phi = self.rng.uniform(0, 2*np.pi, N)

        # 4. Calculate Weights
        # W = (Total Area) / (N * Gaussian_Intensity_at_point)
        # The 2*pi comes from integrating out phi
        gaussian_at_sample = np.exp(-0.5 * (theta / sigma)**2)
        d_omega = (2 * np.pi * Z) / (N * gaussian_at_sample)
        
        # 5. Convert to Vector
        st = np.sin(theta)
        points_local = np.column_stack((st*np.cos(phi), st*np.sin(phi), np.cos(theta)))
        
        return self._align_to_axis(points_local, beam_axis), d_omega


    def _align_to_axis(self, points: np.ndarray, target_axis: Tuple) -> np.ndarray:
        """Rotates points from local Z-aligned frame to target_axis."""
        if target_axis is None: 
            return points
            
        target = np.array(target_axis)
        norm = np.linalg.norm(target)
        if norm == 0:
            return points
        target = target / norm
        
        z_hat = np.array([0.0, 0.0, 1.0])
        
        # Dot product to check alignment
        c = np.dot(z_hat, target)
        
        # Case 1: Already aligned
        if c > 0.999999:
            return points
            
        # Case 2: Opposite direction
        if c < -0.999999:
            # Flip Z and X (to preserve handedness, effectively 180 deg rotation)
            points_copy = points.copy()
            points_copy[:, 2] *= -1 
            points_copy[:, 0] *= -1 
            return points_copy
            
        # Case 3: Rodrigues rotation
        v = np.cross(z_hat, target)
        s_sq = np.dot(v, v)
        
        # Skew-symmetric cross-product matrix
        vx = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
        
        R = np.eye(3) + vx + (vx @ vx) * ((1 - c) / s_sq)
        
        # Apply rotation: (N,3) @ (3,3).T
        return points @ R.T

    def _transverse_basis_batch_rod(self, ks: np.ndarray, beam_axis: Tuple) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes basis vectors e1, e2 orthogonal to K.
        Aligns e1/e2 roughly with global coordinates relative to beam_axis.
        """
        beam_axis = np.array(beam_axis)
        beam_norm = np.linalg.norm(beam_axis)
        if beam_norm > 0:
            n = beam_axis / beam_norm
        else:
            n = np.array([0., 0., 1.])
            
        ks_norm = ks / np.linalg.norm(ks, axis=1, keepdims=True)
        
        # Choose a stable reference vector 'u' orthogonal to 'n'
        if np.abs(n[2]) < 0.9: 
            u = np.cross(n, [0.0, 0.0, 1.0])
        else: 
            u = np.cross(n, [0.0, 1.0, 0.0])
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        
        # Rotate u, v to be orthogonal to specific k vectors (Rodrigues again)
        # Axis of rotation is cross(n, k)
        w = np.cross(n, ks_norm)
        s = np.linalg.norm(w, axis=1, keepdims=True)
        c = np.sum(n * ks_norm, axis=1, keepdims=True)
        
        e1 = np.tile(u, (len(ks), 1))
        e2 = np.tile(v, (len(ks), 1))
        
        mask = s[:, 0] > 1e-9
        if np.any(mask):
            wn = w[mask] / s[mask]
            
            # Rodrigues formula applied to vectors u and v
            # v_rot = v*cos + (k x v)*sin + k*(k.v)*(1-cos)
            
            # Precompute dot terms
            u_dot = np.sum(wn * u, axis=1, keepdims=True)
            v_dot = np.sum(wn * v, axis=1, keepdims=True)
            
            e1[mask] = (u * c[mask] + 
                        np.cross(wn, u) * s[mask] + 
                        wn * u_dot * (1 - c[mask]))
            
            e2[mask] = (v * c[mask] + 
                        np.cross(wn, v) * s[mask] + 
                        wn * v_dot * (1 - c[mask]))
            
        return e1, e2