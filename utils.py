"""
Utility Functions
=================

Miscellaneous helpers and validation functions.
"""

import numpy as np

def get_stokes_parameters(E, normalize=False):
    """
    Computes Stokes parameters from a complex electric field vector E.
    (xy plane is transverse plane)
    
    Args:
        E: Numpy array of shape (3, ...) representing (Ex, Ey, Ez).
        normalize: If True, returns s1, s2, s3 normalized by S0.
    
    Returns:
        Dictionary {'S0': ..., 'S1': ..., 'S2': ..., 'S3': ...}
        and {'S0': ..., 's1': ..., 's2': ..., 's3': ...} if normalized
    """
    # Extract Transverse Components
    Ex = E[0]
    Ey = E[1]
    
    # Conjugates
    Ex_star = np.conj(Ex)
    Ey_star = np.conj(Ey)
    
    # Calculate Unnormalized Stokes
    I_x = np.real(Ex * Ex_star)
    I_y = np.real(Ey * Ey_star)
    
    S0 = I_x + I_y
    S1 = I_x - I_y
    S2 = 2 * np.real(Ex * Ey_star)
    S3 = 2 * np.imag(Ex * Ey_star)
    
    if normalize:
        # Avoid division by zero
        denom = S0.copy()
        denom[denom == 0] = 1.0
        return {'S0': S0, 's1': S1/denom, 's2': S2/denom, 's3': S3/denom}
        
    return {'S0': S0, 'S1': S1, 'S2': S2, 'S3': S3}

def get_ellipse_parameters(E):
    """
    Calculates the geometric properties of the polarization ellipse.
    Useful for plotting.
    (xy plane is transverse plane)

    Args:
        E: Numpy array of shape (3, ...).
        
    Returns:
        Dictionary containing:
        - 'psi': Orientation angle [-pi/2, pi/2]
        - 'chi': Ellipticity angle [-pi/4, pi/4]
        - 'delta': Phase difference (phi_y - phi_x)
        - 'a': Semi-major axis length (normalized to local intensity)
        - 'b': Semi-minor axis length (normalized to local intensity)
        - 'handedness': +1 for LCP/CCW, -1 for RCP/CW (based on S3 sign)
    """
    # Get Stokes
    stokes = get_stokes_parameters(E, normalize=False)
    S0, S1, S2, S3 = stokes['S0'], stokes['S1'], stokes['S2'], stokes['S3']
    
    # 1. Orientation Angle (Psi)
    psi = 0.5 * np.arctan2(S2, S1)
    
    # 2. Ellipticity Angle (Chi)
    s3 = S3 / (S0 + 1e-15) 
    s3 = np.clip(s3, -1.0, 1.0)
    chi = 0.5 * np.arcsin(s3)
    
    # 3. Phase Difference (Delta)
    # angle(Ey) - angle(Ex)
    delta = np.angle(E[1]) - np.angle(E[0])
    
    # 4. Geometry for Plotting (Semi-axes)
    # a is proportional to cos(chi), b is proportional to sin(chi)
    # Scaled by amplitude sqrt(S0)
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

def decompose_in_polarization_basis(E, u):
    """
    Decompose a transverse electric field into an orthonormal
    polarization basis defined by vector u
    (xy plane is transverse plane)

    Args:
        E: field array (3, ...)
        u: reference polarization vector (2,)
    (xy plane is transverse plane)
    Returns:
        {'E_u', 'E_v', 'u_hat', 'v_hat'}
    """
    # Transverse components
    Ex, Ey, _ = E

    # Normalize first basis vector
    u = np.asarray(u, dtype=complex)
    u_hat = u / np.linalg.norm(u)

    # Second orthonormal basis vector
    v_hat = np.array([-np.conj(u_hat[1]), np.conj(u_hat[0])],dtype=complex)

    # Project field (Hermitian inner product)
    E_u = np.conj(u_hat[0]) * Ex + np.conj(u_hat[1]) * Ey
    E_v = np.conj(v_hat[0]) * Ex + np.conj(v_hat[1]) * Ey

    return {
        'E_u': E_u,
        'E_v': E_v,
        'u_hat': u_hat,
        'v_hat': v_hat
    }
