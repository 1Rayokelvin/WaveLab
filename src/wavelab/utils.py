"""
Utility Functions
=================

Miscellaneous helper functions for calculating optical properties,
polarization ellipses, and basis transformations.
"""
from typing import Dict, Union, Tuple
import numpy as np

def get_stokes_params(E1: np.ndarray, E2: np.ndarray, normalize: bool = False) -> Dict[str, np.ndarray]:
    """
    Computes Stokes parameters from two orthogonal electric field components.
    
    For a beam propagating along the Z-axis, you can pass `FieldResult.E[0]` (Ex) 
    and `FieldResult.E[1]` (Ey) to calculate the transverse Stokes parameters.

    Parameters
    ----------
    E1, E2 : np.ndarray
        Complex electric field components of the same shape.
    normalize : bool, optional
        If True, returns the normalized Stokes parameters (s1, s2, s3) where 
        s_i = S_i / S0. Defaults to False.

    Returns
    -------
    dict 
        - If normalize=True:  {'S0': ..., 's1': ..., 's2': ..., 's3': ...}
        - If normalize=False: {'S0': ..., 'S1': ..., 'S2': ..., 'S3': ...}
    """    
    I1 = np.abs(E1)**2
    I2 = np.abs(E2)**2
    
    S0 = I1 + I2
    S1 = I1 - I2
    
    # Calculate cross term
    cross_term = E1 * np.conj(E2)
    S2 = 2 * np.real(cross_term)
    S3 = 2 * np.imag(cross_term)
    
    if normalize:
        denom = np.where(S0 == 0, 1.0, S0)
        return {'S0': S0, 's1': S1/denom, 's2': S2/denom, 's3': S3/denom}
        
    return {'S0': S0, 'S1': S1, 'S2': S2, 'S3': S3}

def get_pol_ellipse_params(E1: np.ndarray, E2: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Calculates the geometric properties of the polarization ellipse
    formed by two orthogonal electric field components.

    Parameters
    ----------
    E1, E2 : np.ndarray
        Complex electric field components of the same shape.
        
    Returns
    -------
    dict 
        Dictionary containing arrays of the same shape as inputs:
        - 'psi': Orientation angle in[-pi/2, pi/2]
        - 'chi': Ellipticity angle in [-pi/4, pi/4]
        - 'delta': Phase difference (phi_2 - phi_1) in range [-pi, pi]
        - 'a': Semi-major axis length (normalized to local intensity)
        - 'b': Semi-minor axis length (normalized to local intensity)
        - 'handedness': +1 for LCP/CCW, -1 for RCP/CW (based on S3 sign)
    """
    stokes = get_stokes_params(E1, E2, normalize=True)
    S0, s1, s2, s3 = stokes['S0'], stokes['s1'], stokes['s2'], stokes['s3']
    
    # 1. Orientation Angle (Psi)
    psi = 0.5 * np.arctan2(s2, s1)
    
    # 2. Ellipticity Angle (Chi)
    s3_norm = np.clip(s3, -1.0, 1.0)
    chi = 0.5 * np.arcsin(s3_norm)
    
    # 3. Phase Difference (Delta)
    delta = np.angle(E2 * np.conj(E1))
    
    # 4. Geometry for Plotting (Semi-axes)
    amp = np.sqrt(S0)
    a = amp * np.cos(chi)
    b = amp * np.abs(np.sin(chi))
    
    return {
        'psi': psi, 
        'chi': chi, 
        'delta': delta,
        'a': a, 
        'b': b,
        'handedness': np.sign(s3)
    }

def decompose_in_basis(E1: np.ndarray, E2: np.ndarray, u: Union[Tuple[complex, complex], np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Decomposes a 2-component complex field (E1, E2) into an orthonormal 
    polarization basis defined by the reference vector `u`.

    Parameters
    ----------
    E1, E2 : np.ndarray
        Complex arrays representing the field in the original basis.
    u : tuple or np.ndarray
        Reference vector (2,) defining the first new basis vector (in the same plane).
        
    Returns
    -------
    dict 
        Contains the new basis vectors and the projected field components:
        {'u_hat': array, 'v_hat': array, 'E_u': array, 'E_v': array}
    """
    u_arr = np.asarray(u, dtype=complex)
    norm = np.linalg.norm(u_arr)
    u_hat = u_arr / norm if norm > 0 else u_arr
    
    # Construct orthogonal vector v_hat
    v_hat = np.array([-np.conj(u_hat[1]), np.conj(u_hat[0])], dtype=complex)

    E_u = np.conj(u_hat[0]) * E1 + np.conj(u_hat[1]) * E2
    E_v = np.conj(v_hat[0]) * E1 + np.conj(v_hat[1]) * E2

    return {
        "E_u": E_u,
        "E_v": E_v,
        "u_hat": u_hat,
        "v_hat": v_hat,
    }