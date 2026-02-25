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
    Implementations of Spatial Profiles in k-space.
    
    All functions here take a k-vector array (shape: 3 x N) and spatial parameters, 
    returning the complex amplitude of the field in k-space.
    """

    @staticmethod
    def uniform(k_vec: np.ndarray, **kwargs) -> np.ndarray:
        """
        Flat angular spectrum (Uniform k-space).
        
        Weights all plane wave directions equally. Physically, if the modes are 
        coherent (locked phase), this corresponds to an infinitesimally small 
        point source or a perfectly spherical converging wave. If phases are 
        random, this simulates isotropic diffuse light.

        Formula:
            S(k) = 1.0 + 0j

        Parameters
        ----------
        k_vec : np.ndarray
            Array of k-vectors of shape (3, N) or (3,).
            
        Returns
        -------
        np.ndarray
            Array of ones matching the length of the input k-vectors.
        """
        if k_vec.ndim == 1:
            return 1.0 + 0j
        return np.ones(k_vec.shape[1], dtype=complex)

    @staticmethod
    def gaussian(k_vec: np.ndarray, sigma_k_perp: float) -> np.ndarray:
        """
        Gaussian spatial profile in k-space.

        Mathematically defines a fundamental Gaussian beam (TEM00).

        Formula:
            S(kx, ky) = exp( -(kx^2 + ky^2) / (2 * sigma_k_perp^2) )

        Parameters
        ----------
        k_vec : np.ndarray
            Array of k-vectors of shape (3, N) where rows are kx, ky, kz.
        sigma_k_perp : float
            The standard deviation of the Gaussian envelope in k-space.
            This is inversely proportional to the real-space beam waist (w0).

        Returns
        -------
        np.ndarray
            Complex amplitude weights for each plane wave mode.
        """
        kx, ky = k_vec[0], k_vec[1]
        gaussian_profile = np.exp(-(kx**2 + ky**2) / (2 * sigma_k_perp**2))
        return gaussian_profile.astype(complex)

    @staticmethod
    def tophat(k_vec: np.ndarray, k_perp_max: float) -> np.ndarray:
        """
        Uniform Top-Hat disk spectrum.

        Defines a sharp cutoff in transverse momentum. In real space, 
        this produces a J1(r)/r (Airy disk) transverse profile.

        Formula:
            S(kx, ky) = 1.0   if (kx^2 + ky^2) <= k_perp_max^2
            S(kx, ky) = 0.0   otherwise

        Parameters
        ----------
        k_vec : np.ndarray
            Array of k-vectors of shape (3, N).
        k_perp_max : float
            The maximum transverse spatial frequency allowed.

        Returns
        -------
        np.ndarray
            Complex amplitude weights (1.0 or 0.0).
        """
        mask = (k_vec[0]**2 + k_vec[1]**2) < k_perp_max**2
        return mask.astype(complex)

    @staticmethod
    def laguerre_gauss(k_vec: np.ndarray, p: int, l: int, sigma_k_perp: float) -> np.ndarray:
        """
        Angular spectrum for Laguerre-Gauss (Vortex) modes.

        Defines a beam carrying Orbital Angular Momentum (OAM). 
        The parameter `l` defines the topological charge, and `p` defines 
        the number of radial nodes.

        Formula:
            rho^2 = kx^2 + ky^2
            phi_k = arctan2(ky, kx)
            X = rho^2 / sigma_k_perp^2
            
            S(kx, ky) = X^(|l|/2) * L_p^{|l|}(X) * exp(-X/2) * exp(i * l * phi_k)

        Parameters
        ----------
        k_vec : np.ndarray
            Array of k-vectors of shape (3, N).
        p : int
            Radial index (p >= 0). Determines the number of radial rings.
        l : int
            Azimuthal index (topological charge). Determines OAM.
        sigma_k_perp : float
            Transverse scaling parameter in k-space.

        Returns
        -------
        np.ndarray
            Complex amplitude weights defining the LG mode.
        """
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
        """
        Angular spectrum for Hermite-Gauss modes.

        Defines higher-order transverse modes in Cartesian coordinates, 
        often seen emitted by laser cavities breaking cylindrical symmetry.

        Formula:
            scale = sqrt(2) / sigma_k_perp
            S(kx, ky) = H_l(kx * scale) * H_m(ky * scale) * exp(-(kx^2 + ky^2) / 2 sigma_k_perp^2)

        Parameters
        ----------
        k_vec : np.ndarray
            Array of k-vectors of shape (3, N).
        l : int
            Transverse mode index in the x-direction (l >= 0).
        m : int
            Transverse mode index in the y-direction (m >= 0).
        sigma_k_perp : float
            Transverse scaling parameter in k-space.

        Returns
        -------
        np.ndarray
            Complex amplitude weights defining the HG mode.
        """
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

        Defined by a Gaussian ring of illumination on the angular hemisphere.
        Creates a "non-diffracting" Bessel beam core in real space, enveloped
        by a Gaussian bounds.

        Formula:
            theta_k = arccos(kz / |k|)
            phi_k = arctan2(ky, kx)
            
            S(k) = exp( -(theta_k - theta_0)^2 / (2 * sigma_theta^2) ) * exp(i * l * phi_k)

        Parameters
        ----------
        k_vec : np.ndarray
            Array of k-vectors of shape (3, N) where rows are kx, ky, kz.
        theta_0 : float
            Cone opening angle (in radians) of the Bessel beam in k-space.
        sigma_theta : float
            Angular thickness (in radians) of the Gaussian ring.
        l : int, optional
            Topological charge. If l != 0, creates a Higher-Order Bessel beam (vortex).

        Returns
        -------
        np.ndarray
            Complex amplitude weights defining the BG mode.
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
    def uniform(wavelength: float, **kwargs) -> float:
        """
        Flat spectral envelope. 

        Formula:
            F(lambda) = 1.0

        Parameters
        ----------
        wavelength : float
            The wavelength to evaluate.
            
        Returns
        -------
        float
            Weight of 1.0 for all inputs.
        """
        return 1.0

    @staticmethod
    def gaussian(wl: float, center: float, sigma: float) -> float:
        """
        Gaussian spectral weight distribution.
        
        Formula:
            F(lambda) = exp( -(lambda - center)^2 / (2 * sigma^2) )

        Parameters
        ----------
        wl : float
            The wavelength to evaluate.
        center : float
            The central wavelength (mean).
        sigma : float
            The spectral bandwidth (standard deviation).

        Returns
        -------
        float
            Real scalar weight for the given wavelength.
        """
        return float(np.exp(-(wl - center)**2 / (2 * sigma**2)))

    @staticmethod
    def lorentzian(wl: float, center: float, gamma: float) -> float:
        """
        Lorentzian spectral weight distribution.
        
        Often models natural line-broadening in atomic emission.

        Formula:
            F(lambda) = gamma^2 / ( (lambda - center)^2 + gamma^2 )

        Parameters
        ----------
        wl : float
            The wavelength to evaluate.
        center : float
            The central resonance wavelength.
        gamma : float
            The Half-Width at Half-Maximum (HWHM) of the spectrum.

        Returns
        -------
        float
            Real scalar weight for the given wavelength.
        """
        return float(gamma**2 / ((wl - center)**2 + gamma**2))

    @staticmethod
    def tophat(wl: float, center: float, width: float) -> float:
        """
        Top-hat (Rectangular) spectral weight distribution.

        Idealized bandpass filter.

        Formula:
            F(lambda) = 1.0  if |lambda - center| <= width / 2
            F(lambda) = 0.0  otherwise

        Parameters
        ----------
        wl : float
            The wavelength to evaluate.
        center : float
            The central wavelength of the bandpass.
        width : float
            The total spectral width of the bandpass.

        Returns
        -------
        float
            Real scalar weight (1.0 or 0.0).
        """
        return 1.0 if abs(wl - center) <= width / 2 else 0.0