"""
spectras.py
========================

Mathematical implementations of spatial (k-space) and spectral (wavelength) distributions.
"""
import numpy as np
from typing import Dict, Any

try:
    from scipy.special import eval_hermite, eval_genlaguerre
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

def _check_scipy():
    if not HAS_SCIPY:
        raise ImportError("Spatial spectra functions require 'scipy'. Please `pip install scipy`.")

class spatial_spectras:
    """
    Implementations of Transverse Spatial Profiles in k-space.
    
    All functions here take a k-vector array and spatial parameters, 
    returning the complex amplitude of the field in k-space.
    """

    @staticmethod
    def gaussian(k_vec: np.ndarray, sigma_k_perp: float) -> np.ndarray:
        """Gaussian spectrum."""
        kx, ky = k_vec[0], k_vec[1]
        gaussian_profile = np.exp(-(kx**2 + ky**2) / (2 * sigma_k_perp**2))
        return gaussian_profile.astype(complex)

    @staticmethod
    def tophat(k_vec: np.ndarray, k_perp_max: float) -> np.ndarray:
        """Uniform Top-Hat disk spectrum."""
        mask = (k_vec[0]**2 + k_vec[1]**2) < k_perp_max**2
        return mask.astype(complex)

    @staticmethod
    def laguerre_gauss(k_vec: np.ndarray, p: int, l: int, sigma_k_perp: float) -> np.ndarray:
        """Spectrum for Laguerre-Gauss (vortex) modes."""
        _check_scipy()
        kx, ky = k_vec[0], k_vec[1]
        k_perp_sq = kx**2 + ky**2
        phi = np.arctan2(ky, kx)
        
        X = k_perp_sq / (sigma_k_perp**2)
        poly_val = eval_genlaguerre(p, abs(l), X)
        gaussian_env = np.exp(-0.5 * X)
        radial = (np.sqrt(X)) ** abs(l)
        vortex = np.exp(1j * l * phi)
    
        res = (radial * poly_val * gaussian_env * vortex).astype(complex)
        res = np.where(k_perp_sq == 0, 0j, res)
        return res if res.size > 1 else res[0]
    
    @staticmethod
    def hermite_gauss(k_vec: np.ndarray, l: int, m: int, sigma_k_perp: float) -> np.ndarray:
        """Angular spectrum for Hermite-Gauss modes."""
        _check_scipy()
        kx, ky = k_vec[0], k_vec[1]
        
        # Dimensionless scaling: x / (w0_k / sqrt(2)) -> x * sqrt(2) / sigma
        scale = np.sqrt(2) / sigma_k_perp
        
        term_x = eval_hermite(l, kx * scale)
        term_y = eval_hermite(m, ky * scale)
        gaussian = np.exp(-(kx**2 + ky**2) / (2 * sigma_k_perp**2))
        
        res = (term_x * term_y * gaussian).astype(complex)
        return res

    @staticmethod
    def bessel_gauss(k_vec: np.ndarray, theta_0: float, sigma_theta: float, l: int = 0) -> np.ndarray:
        """
        Angular spectrum for a Bessel-Gauss beam.
        """
        # Extract components - each has shape (N,)
        kx, ky, kz = k_vec[0], k_vec[1], k_vec[2]
        k_mag = np.sqrt(kx**2 + ky**2 + kz**2)
        res = np.zeros_like(k_mag, dtype=complex)
        mask = k_mag > 0
        
        theta = np.zeros_like(k_mag)
        theta[mask] = np.arccos(np.clip(kz[mask] / k_mag[mask], -1, 1))
        
        diff = theta - theta_0
        ring_profile = np.exp(-(diff**2) / (2 * sigma_theta**2))
        
        if l != 0:
            phi = np.arctan2(ky, kx)  # shape (N,)
            vortex = np.exp(1j * l * phi)  # shape (N,)
            res[mask] = (ring_profile[mask] * vortex[mask])
        else:
            res[mask] = ring_profile[mask]
        
        return res

class spectral_spectras:
    """
    Implementations of Longitudinal Spectral Profiles (Polychromatic Envelopes).
    
    All functions here take a wavelength and spectral parameters, 
    returning a real scalar weight for that specific wavelength.
    """

    @staticmethod
    def gaussian(wl: float, center: float, sigma: float) -> float:
        """Gaussian spectral weight distribution."""
        return float(np.exp(-(wl - center)**2 / (2 * sigma**2)))

    @staticmethod
    def lorentzian(wl: float, center: float, gamma: float) -> float:
        """Lorentzian spectral weight distribution."""
        return float(gamma**2 / ((wl - center)**2 + gamma**2))

    @staticmethod
    def tophat(wl: float, center: float, width: float) -> float:
        """Top-hat (Rectangular) spectral weight distribution."""
        return 1.0 if abs(wl - center) <= width / 2 else 0.0