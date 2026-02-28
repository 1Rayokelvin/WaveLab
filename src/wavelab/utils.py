"""
Utility Functions
=================

Miscellaneous helper functions.
"""
import numpy as np

def get_stokes_params(E1, E2, normalize=False):
    """
    Computes Stokes parameters from two orthogonal electric field components.
    
    Parameters
    ----------
        - E1, E2 : Complex electric field components of same shape.
        - normalize: If True, returns s1, s2, s3 instead of S1,S2,S3.

    Returns
    -------
    dict 
        - if normalize is true, {'S0': ..., 's1': ..., 's2': ..., 's3': ...}
        - otherwise, {'S0': ..., 'S1': ..., 'S2': ..., 'S3': ...} .
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

def get_pol_ellipse_params(E1, E2):
    """
    Calculates the geometric properties of the polarization ellipse
    formed by two orthogonal electric field components.

    Parameters
    ----------
        E1, E2 : Complex electric field components of same shape.
        
    Returns
    -------
    dict 
        keys are:
        - 'psi': Orientation angle [-pi/2, pi/2]
        - 'chi': Ellipticity angle [-pi/4, pi/4]
        - 'delta': Phase difference (phi_2 - phi_1) in range [-pi, pi]
        - 'a': Semi-major axis length (normalized to local intensity)
        - 'b': Semi-minor axis length (normalized to local intensity)
        - 'handedness': +1 for LCP/CCW, -1 for RCP/CW (based on S3 sign)
    """
    stokes = get_stokes_params(E1, E2)
    S0, S1, S2, S3 = stokes['S0'], stokes['S1'], stokes['S2'], stokes['S3']
    
    # 1. Orientation Angle (Psi)
    psi = 0.5 * np.arctan2(S2, S1)
    
    # 2. Ellipticity Angle (Chi)
    denom = np.where(S0 == 0, 1.0, S0)
    s3_norm = np.clip(S3 / denom, -1.0, 1.0)
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
        'handedness': np.sign(S3)
    }

def decompose_in_basis(E1, E2, u):
    """
    Decompose a 2-component complex field (E1, E2)
    into an orthonormal polarization basis defined by u.

    Parameters
    ----------
        E1, E2 : complex arrays representing the field in the original basis
        u      : reference vector (2,) defining first new basis vector (in same plane!)
        
    Returns
    -------
    dict 
        {'u_hat', 'v_hat', 'E_u', 'E_v' }.
    """
    u = np.asarray(u, dtype=complex)
    u_hat = u / np.linalg.norm(u)

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