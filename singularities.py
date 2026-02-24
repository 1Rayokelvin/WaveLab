"""
Topological Singularity Detection
==================================

Find and analyze polarization singularities in 3D electromagnetic fields.
"""

import numpy as np
from numpy import real, imag, conj, sign, dot

class SingularityFinder:
    """
    Locates and traces electric field singularities in 3D space.
    
    Works with a physics engine to find points where field properties 
    become degenerate (C-points, C^T-points, L^T-points) and trace their evolution 
    through space as lines or loops.
    
    Time t=0 by default as polarization is time independent in monochromatic fields. 
    Use self.t to change this behaviour.
    
    Parameters
    ----------
    physics_engine : FieldEngine
        An initialized FieldEngine capable of evaluating points in space.
    requested_backend : str, optional
        Computation backend to use for point queries ('numpy', 'numba', 'auto'). 
        Default is 'numba'.
    
    Example
    -------
    ```
    expt = setup_experiment(cfg)
    E, _, _ = expt.compute_on_op(z=0.0)
    
    finder = singularity_finder(expt)
    pts = finder.find_stokes_C_points(z_val=0.0, E_grid=E)
    ```
    """
    def __init__(self, physics_engine, requested_backend='numba'):
        self.engine = physics_engine
        self.backend_name = self.engine.selector(requested_backend)
        if self.backend_name == 'numpy' and self.engine.config.verbose:
            print("Using numpy for singularity finding.")
            print("Use numba for significant speed boost.")
        self.x = physics_engine.x
        self.y = physics_engine.y
        self.t = 0 
        self.x_min, self.x_max = min(self.x), max(self.x)
        self.y_min, self.y_max = min(self.y), max(self.y)

    # --- Internals ---
    def _compute_field_and_jacobian_at_point(self, x, y, z):
        """Wraps the field_engine's calculator to get data for a single point."""
        E, derivs,_ = self.engine.compute_point(
            x,y,z,t=self.t, need_b=False, backend_name=self.backend_name
            )
        J = np.column_stack(derivs)
        return E, J

    def _newton_raphson_2d(self, value_and_corrector, x0, y0, max_iter=10, tol=1e-6, value_tol=1e-6):
        x, y = x0, y0
        for _ in range(max_iter):
            v, C = value_and_corrector(x, y)
            try:
                delta = np.linalg.solve(C, v)
                x_new, y_new = x - delta[0], y - delta[1]
                if np.hypot(x_new - x, y_new - y) < tol: return x_new, y_new, True
                x, y = x_new, y_new
            except np.linalg.LinAlgError: return x, y, False
        final_v, _ = value_and_corrector(x, y)
        if np.linalg.norm(final_v) < value_tol: return x, y, True
        return x, y, False

    def _zero_cross_mask(self, field_signs):
        cell_max = np.maximum.reduce([field_signs[:-1, :-1], field_signs[:-1, 1:], 
                                      field_signs[1:, :-1], field_signs[1:, 1:]])
        cell_min = np.minimum.reduce([field_signs[:-1, :-1], field_signs[:-1, 1:], 
                                      field_signs[1:, :-1], field_signs[1:, 1:]])
        return (cell_max > 0) & (cell_min < 0)

    def _plane_fit_zero_guess(self, y_idx, x_idx, data_q1, data_q2):
        q1_cell = data_q1[y_idx:y_idx+2, x_idx:x_idx+2].flatten()
        q2_cell = data_q2[y_idx:y_idx+2, x_idx:x_idx+2].flatten()
        coords = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        fit_matrix = np.hstack([coords, np.ones((4, 1))])
        try:
            p1 = np.linalg.lstsq(fit_matrix, q1_cell, rcond=None)[0]
            p2 = np.linalg.lstsq(fit_matrix, q2_cell, rcond=None)[0]
            A = np.array([[p1[1], p1[0]], [p2[1], p2[0]]])
            b = -np.array([p1[2], p2[2]])
            dx, dy = np.linalg.solve(A, b)
            return dx, dy, True
        except np.linalg.LinAlgError: return 0, 0, False

    def _find_singularities_template(self, z_value, candidate_coords, initial_guess_func, value_and_corrector_func, 
                                     max_iter, tol, value_tol, x_bounds=None, y_bounds=None):
        found_points = []
        for y_idx, x_idx in candidate_coords:
            dx, dy, success = initial_guess_func(y_idx, x_idx)
            if not success or not (0 <= dx < 1 and 0 <= dy < 1): continue
            continuous_x = self.x[x_idx] + dx * (self.x[x_idx+1] - self.x[x_idx])
            continuous_y = self.y[y_idx] + dy * (self.y[y_idx+1] - self.y[y_idx])
            x_final, y_final, confident = self._newton_raphson_2d(lambda x_, y_: value_and_corrector_func(x_, y_, z_value),
                                                                    continuous_x, continuous_y, max_iter=max_iter, tol=tol, value_tol=value_tol)
            if x_bounds and not (x_bounds[0] <= x_final <= x_bounds[1]): confident = False
            if y_bounds and not (y_bounds[0] <= y_final <= y_bounds[1]): confident = False
            found_points.append({'position': (x_final, y_final, z_value), 'guess': (continuous_x, continuous_y, z_value), 'confident': confident})
        return found_points
    
    # --- Point finding methods ---
    def _stokes_and_grads_from_EJ(self, E, J):
        Ex, Ey = E[0], E[1]
        Ex_x, Ex_y, Ex_z = J[0]
        Ey_x, Ey_y, Ey_z = J[1]

        S0 = abs(Ex)**2 + abs(Ey)**2
        S1 = abs(Ex)**2 - abs(Ey)**2
        S2 = 2 * np.real(Ex * np.conj(Ey))
        S3 = 2 * np.imag(Ex * np.conj(Ey))

        S0_x = 2 * np.real(Ex_x * np.conj(Ex) + Ey_x * np.conj(Ey))
        S0_y = 2 * np.real(Ex_y * np.conj(Ex) + Ey_y * np.conj(Ey))
        S0_z = 2 * np.real(Ex_z * np.conj(Ex) + Ey_z * np.conj(Ey))

        S1_x = 2 * np.real(Ex_x * np.conj(Ex)) - 2 * np.real(Ey_x * np.conj(Ey))
        S1_y = 2 * np.real(Ex_y * np.conj(Ex)) - 2 * np.real(Ey_y * np.conj(Ey))
        S1_z = 2 * np.real(Ex_z * np.conj(Ex)) - 2 * np.real(Ey_z * np.conj(Ey))

        S2_x = 2 * np.real(Ex_x * np.conj(Ey) + Ex * np.conj(Ey_x))
        S2_y = 2 * np.real(Ex_y * np.conj(Ey) + Ex * np.conj(Ey_y))
        S2_z = 2 * np.real(Ex_z * np.conj(Ey) + Ex * np.conj(Ey_z))

        return (
            (S0, S1, S2, S3),
            np.array([S0_x, S0_y, S0_z]),
            np.array([S1_x, S1_y, S1_z]),
            np.array([S2_x, S2_y, S2_z]),
        )
    
    def _stokes_c_point_value_and_corrector(self, x, y, z):
        E, J = self._compute_field_and_jacobian_at_point(x, y, z)

        (S0, S1, S2, _), grad_S0, grad_S1, grad_S2 = \
            self._stokes_and_grads_from_EJ(E, J)

        S0_safe = S0 if S0 > 1e-12 else 1.0

        f_sp = np.array([S1, S2]) / S0_safe

        J_sp = np.array([
            (S0 * grad_S1[:2] - S1 * grad_S0[:2]) / S0_safe**2,
            (S0 * grad_S2[:2] - S2 * grad_S0[:2]) / S0_safe**2,
        ])

        return f_sp, J_sp

    def find_stokes_C_points(self, z_value, E_grid, max_iter=10, pos_tol=1e-6, value_tol=1e-6):
        """
        Finds Stokes C-points, where polarization (2D) is purely circular (s1=s2=0).

        Returns list of dicts with 'position', 'guess', 'type' (Star/Lemon/Monstar), 
        'intensity', 'handedness', and 'confident'.

        Args:
            z_value: z-coordinate of observation plane
            E_grid: Electric field array (3, ny, nx) from engine
            max_iter: Newton-Raphson iterations per candidate
            pos_tol: Position convergence tolerance
            value_tol: Residual tolerance for s1, s2
        """
        Ex, Ey, _ = E_grid
        S0 = abs(Ex)**2 + abs(Ey)**2; S0[S0 == 0] = 1
        s1 = (abs(Ex)**2 - abs(Ey)**2) / S0
        s2 = (2 * real(Ex * conj(Ey))) / S0
        candidate_coords = np.argwhere(self._zero_cross_mask(sign(s1)) & self._zero_cross_mask(sign(s2)))
        def guess_func(y_idx, x_idx): return self._plane_fit_zero_guess(y_idx, x_idx, s1, s2)
        found_points = self._find_singularities_template(
            z_value, candidate_coords, guess_func, self._stokes_c_point_value_and_corrector, 
            max_iter, pos_tol, value_tol, (min(self.x),max(self.x)), (min(self.y),max(self.y))
            )
        for point in found_points:
            x_val, y_val, _ = point['position']
            E_pt, J_pt = self._compute_field_and_jacobian_at_point(x_val, y_val, z_value)
            all_S, _, S1_derivs, S2_derivs = \
                self._stokes_and_grads_from_EJ(E_pt, J_pt)
            
            S0_val = all_S[0]; S0_safe = S0_val if S0_val > 1e-12 else 1.0
            S1_x, S1_y = S1_derivs[0]/S0_safe, S1_derivs[1]/S0_safe
            S2_x, S2_y = S2_derivs[0]/S0_safe, S2_derivs[1]/S0_safe
            
            D_I = S1_x * S2_y - S1_y * S2_x
            if D_I < 0:
                point['type'] = 'Star'
            else:
                fact1 = (2*S1_y + S2_x)**2 - 3*S2_y*(2*S1_x - S2_y)
                fact2 = (2*S1_x - S2_y)**2 + 3*S2_x*(2*S1_y + S2_x)
                term2 = (2*S1_x*S1_y+ S1_x*S2_x- S1_y*S2_y+ 4*S2_x*S2_y)**2

                if fact1 * fact2 - term2 < 0:
                    point['type'] = 'Lemon'
                else:
                    point['type'] = 'Monstar'

            point['intensity'] = all_S[0]; point['handedness'] = sign(all_S[3])
        return found_points

    def _C_T_points_v_and_c(self, x, y, z_value):
        E, J = self._compute_field_and_jacobian_at_point(x, y, z_value)
        E_x, E_y = J[:, 0], J[:, 1]
        E2 = dot(E, E); dE2_dx = 2 * dot(E, E_x); dE2_dy = 2 * dot(E, E_y)
        f_cp = np.array([real(E2), imag(E2)])
        J_cp = np.array([[real(dE2_dx), real(dE2_dy)], [imag(dE2_dx), imag(dE2_dy)]])
        return f_cp, J_cp

    def find_C_T_points(self, z_value, E_grid, max_iter=10, pos_tol=1e-6, value_tol=1e-6):
        """
        Finds C^T points where true circular polarization (3D) occurs (E·E=0).
        
        Returns list of dicts with 'position', 'guess', and 'confident'.
        
        Args:
            z_value: z-coordinate of observation plane
            E_grid: Electric field array (3, ny, nx) from engine
            max_iter: Newton-Raphson iterations per candidate
            pos_tol: Position convergence tolerance
            value_tol: Residual tolerance for E·E
        """

        E2 = np.sum(E_grid**2, axis=0); re_E2, im_E2 = real(E2), imag(E2)
        candidate_coords = np.argwhere(self._zero_cross_mask(sign(re_E2)) & self._zero_cross_mask(sign(im_E2)))
        def guess_func(y_idx, x_idx): return self._plane_fit_zero_guess(y_idx, x_idx, re_E2, im_E2)
        return self._find_singularities_template(z_value, candidate_coords, guess_func, 
                                        self._C_T_points_v_and_c, max_iter, pos_tol, value_tol)
    
    def _L_T_points_v_and_c(self, x, y, z_value):
        """
        Computes the minimization step for Vector L-points where 
        N = Re(E) x Im(E) = 0.
        
        Since this is an overdetermined system (3 equations, 2 vars),
        we return the Normal Equations (J.T @ J and J.T @ val) 
        compatible with the generic Newton solver.
        """
        # 1. Get Fields and derivatives
        E, J = self._compute_field_and_jacobian_at_point(x, y, z_value)
        
        # J is 3x3 (columns are dx, dy, dz), we need dx and dy columns
        # E_x and E_y are 3-element vectors representing partial derivatives of E vector
        E_x, E_y = J[:, 0], J[:, 1]

        ReE, ImE = real(E), imag(E)
        ReE_x, ImE_x = real(E_x), imag(E_x)
        ReE_y, ImE_y = real(E_y), imag(E_y)

        # 2. Calculate Normal Vector N = Re(E) x Im(E)
        # Target is N = [0, 0, 0]
        n = np.cross(ReE, ImE)

        # 3. Calculate Jacobian of N (dN/dx, dN/dy)
        # Product rule: d(A x B) = dA x B + A x dB
        dn_dx = np.cross(ReE_x, ImE) + np.cross(ReE, ImE_x)
        dn_dy = np.cross(ReE_y, ImE) + np.cross(ReE, ImE_y)
        
        # J_lp is 3x2 matrix
        J_lp = np.column_stack([dn_dx, dn_dy])

        # 4. Formulate Normal Equations for Least Squares
        # The Newton solver solves C * delta = v
        # For minimization of |N|^2, C = J.T @ J, v = J.T @ n
        C = J_lp.T @ J_lp
        v = J_lp.T @ n
        
        return v, C

    def find_L_T_points(self, z_value, E_grid, max_iter=10, pos_tol=1e-6, value_tol=1e-6):
        """
        Finds L^T points where true linear polarization (3D) occurs (Re(E) x Im(E)=0).
        
        Returns list of dicts with 'position', 'guess', and 'confident'.
        
        Args:
            z_value: z-coordinate of observation plane
            E_grid: Electric field array (3, ny, nx) from engine
            max_iter: Newton-Raphson iterations per candidate
            pos_tol: Position convergence tolerance
            value_tol: Residual tolerance for normal vector
        """
        # Compute N grid
        N_e = np.cross(real(E_grid), imag(E_grid), axis=0)

        # Identify candidate cells. 
        mask_x = self._zero_cross_mask(sign(N_e[0]))
        mask_y = self._zero_cross_mask(sign(N_e[1]))
        mask_z = self._zero_cross_mask(sign(N_e[2]))
        
        candidate_coords = np.argwhere(mask_x & mask_y & mask_z)

        # Initial guess function using Least Squares Plane Fit on N_x, N_y, N_z
        def guess_func(y_idx, x_idx):
            # Extract 2x2 cells for all 3 components
            Nx_cell = N_e[0, y_idx:y_idx+2, x_idx:x_idx+2].flatten()
            Ny_cell = N_e[1, y_idx:y_idx+2, x_idx:x_idx+2].flatten()
            Nz_cell = N_e[2, y_idx:y_idx+2, x_idx:x_idx+2].flatten()

            # Design matrix for bilinear/plane fit (y, x, 1)
            # coords corresponds to flattened [0,0], [0,1], [1,0], [1,1]
            coords = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) 
            fit_matrix = np.hstack([coords, np.ones((4, 1))])

            try:
                # Fit planes: N_i = p[0]*y + p[1]*x + p[2]
                px = np.linalg.lstsq(fit_matrix, Nx_cell, rcond=None)[0]
                py = np.linalg.lstsq(fit_matrix, Ny_cell, rcond=None)[0]
                pz = np.linalg.lstsq(fit_matrix, Nz_cell, rcond=None)[0]

                # We want Nx=0, Ny=0, Nz=0. This is 3 eq, 2 unknowns.
                # Solve A * [dy, dx].T = b
                # A columns are [coeff_y, coeff_x] (from p[0], p[1])
                A = np.array([
                    [px[0], px[1]], 
                    [py[0], py[1]], 
                    [pz[0], pz[1]]
                ])
                b = -np.array([px[2], py[2], pz[2]])

                # Least squares solve for the initial sub-pixel guess
                (dy, dx), _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                return dx, dy, True
            except np.linalg.LinAlgError:
                return 0, 0, False

        # Use the template to find and refine points
        return self._find_singularities_template(z_value, candidate_coords, guess_func, 
                                        self._L_T_points_v_and_c, max_iter, pos_tol, value_tol)

    # --- Line tracing methods ---
    def _corrector_pinv_single(self, func_val_jac, x0, y0, z0, max_iter=10, tol=1e-6):
        """
        Refines a single 3D point onto the line defined by func_val_jac(x,y,z) = 0
        using Minimum-Norm (Pseudoinverse) updates.
        """
        x, y, z = x0, y0, z0
        
        for _ in range(max_iter):
            # Get Value (2,) and Jacobian (2, 3)
            vals, J = func_val_jac(x, y, z)
            
            # Check convergence (magnitude of residuals)
            if np.linalg.norm(vals) < tol:
                return x, y, z, True

            try:
                # Moore-Penrose Pseudoinverse for "fat" matrix J (underdetermined system)
                # Delta = J.T * (J * J.T)^-1 * vals
                JJT = np.dot(J, J.T)  # (2, 2)
                lambda_vec = np.linalg.solve(JJT, vals) # (2,)
                delta = np.dot(J.T, lambda_vec) # (3,)
                
                x -= delta[0]
                y -= delta[1]
                z -= delta[2]
            except np.linalg.LinAlgError:
                return x, y, z, False
                
        return x, y, z, False

    def _trace_line_generic(self, starting_point, val_jac_func, ds, max_steps, max_iter, value_tol):
        """
        Generic routine to trace a line given a function that returns (Values, Jacobian_3D).
        """
        trajectory = []
        
        # Unpack start
        if isinstance(starting_point, dict):
            p = starting_point['position']
        else:
            p = starting_point
        x, y, z = p[0], p[1], p[2]

        # 1. Refine seed (Step 0)
        x, y, z, _ = self._corrector_pinv_single(val_jac_func, x, y, z, 
                                                 max_iter=max_iter*2, tol=value_tol)
        trajectory.append(np.array([x, y, z]))

        for _ in range(max_steps):
            # --- Predictor Step ---
            # Calculate Tangent t = Gradient_1 x Gradient_2
            _, J = val_jac_func(x, y, z)
            grad_1 = J[0, :]
            grad_2 = J[1, :]
            
            tangent = np.cross(grad_1, grad_2)
            norm_t = np.linalg.norm(tangent)
            
            if norm_t < 1e-12: break # Singularity or dead end
            tangent /= norm_t
            
            # Move
            x_pred = x + tangent[0] * ds
            y_pred = y + tangent[1] * ds
            z_pred = z + tangent[2] * ds
            
            # --- Corrector Step (Relax onto line) ---
            x, y, z, success = self._corrector_pinv_single(val_jac_func, x_pred, y_pred, z_pred, 
                                                           max_iter=max_iter, tol=value_tol)
            
            if not success: break
            
            # Check bounds if necessary (optional, assumes normalized box 0-1 or similar)
            # if not (x_min < x < x_max ...): break 

            trajectory.append(np.array([x, y, z]))

        return np.array(trajectory)

    def _stokes_C_val_jac_3d(self, x, y, z):
        """Returns [s1, s2] and 2x3 Jacobian for Stokes C-lines."""
        E, J = self._compute_field_and_jacobian_at_point(x, y, z)

        # Use existing helper to get all gradients including Z
        all_S, grad_S0, grad_S1, grad_S2 = self._stokes_and_grads_from_EJ(E, J)
        
        S0, S1, S2 = all_S[0], all_S[1], all_S[2]
        S0_safe = S0 if S0 > 1e-12 else 1.0

        # Normalized Stokes parameters s1 = S1/S0, s2 = S2/S0
        vals = np.array([S1, S2]) / S0_safe
        
        # Quotient Rule for Gradients
        jac = np.array([ 
        (S0 * grad_S1[:] - S1 * grad_S0[:]) / S0_safe**2,
        (S0 * grad_S2[:] - S2 * grad_S0[:]) / S0_safe**2,
            ])
        return vals, jac

    def trace_stokes_C_lines(self, starting_points, ds=0.05, max_steps=500, max_iter=10, value_tol=1e-6):
        """
        Traces Stokes C-lines in 3D from seed points (s1=s2=0 curves).

        Args:
            starting_points: List of seed dicts from find_stokes_C_points() or (x,y,z) tuples
            ds: Step size along tangent to curve
            max_steps: Maximum number of tracing steps
            max_iter: Corrector iterations per step
            value_tol: Residual tolerance for sqrt(s1^2+s2^2)

        Returns:
            List of (N, 3) trajectory arrays
        """
        all_lines = []
        for seed in starting_points:
            line = self._trace_line_generic(
                seed, 
                self._stokes_C_val_jac_3d, ds, 
                max_steps, max_iter, value_tol=value_tol
            )
            all_lines.append(line)
        return all_lines

    def _C_T_val_jac_3d(self, x, y, z):
        """Returns [Re(E^2), Im(E^2)] and 2x3 Jacobian for Vector C^T lines."""
        E, J = self._compute_field_and_jacobian_at_point(x, y, z)
        # E is (3,), J is (3, 3) where columns are dx, dy, dz
        
        # Value: E^2 = E dot E
        E2 = np.dot(E, E)
        vals = np.array([np.real(E2), np.imag(E2)])
        
        # Gradient of E^2: grad(E.E) = 2 * (E . grad(E))
        # This is a tensor contraction.
        # d(E^2)/dx = 2 * (Ex*dEx/dx + Ey*dEy/dx + Ez*dEz/dx)
        # This is equivalent to 2 * (E dot J_col)
        
        grad_psi_x = 2 * np.dot(E, J[:, 0])
        grad_psi_y = 2 * np.dot(E, J[:, 1])
        grad_psi_z = 2 * np.dot(E, J[:, 2])
        
        grad_re = np.array([np.real(grad_psi_x), np.real(grad_psi_y), np.real(grad_psi_z)])
        grad_im = np.array([np.imag(grad_psi_x), np.imag(grad_psi_y), np.imag(grad_psi_z)])
        
        # Stack into 2x3 Jacobian
        jac = np.vstack([grad_re, grad_im])
        
        return vals, jac

    def trace_C_T_lines(self, starting_points, ds=0.05, max_steps=500, max_iter=10, value_tol=1e-6):
        """
        Traces C^T lines in 3D from seed points (E·E=0 curves).
        
        Args:
            starting_points: List of seed dicts from find_C_T_points() or (x,y,z) tuples
            ds: Step size along tangent to curve
            max_steps: Maximum number of tracing steps
            max_iter: Corrector iterations per step
            value_tol: Residual tolerance for E·E
            
        Returns:
            List of (N, 3) trajectory arrays
        """
        all_lines = []
        for seed in starting_points:
            line = self._trace_line_generic(
                seed, 
                self._C_T_val_jac_3d, ds, 
                max_steps, max_iter, value_tol
            )
            all_lines.append(line)
        return all_lines

    def _L_T_val_jac_3d(self, x, y, z):
        """
        Returns values and Jacobian for Vector L-Lines (True Linear Polarization).
        
        Math Note:
        The condition N = Re(E) x Im(E) = 0 is codimension 2.
        The components of N are dependent (N is orthogonal to E). 
        Therefore, vanishing of any two components implies vanishing of the third.
        
        We solve for intersection of N_x = 0 and N_y = 0.
        """
        # 1. Get Fields and derivatives
        E, J = self._compute_field_and_jacobian_at_point(x, y, z)
        
        # J columns are partials: [dE/dx, dE/dy, dE/dz]
        dE_dx, dE_dy, dE_dz = J[:, 0], J[:, 1], J[:, 2]

        ReE, ImE = np.real(E), np.imag(E)
        
        # Real/Imag parts of derivatives
        ReE_x, ImE_x = np.real(dE_dx), np.imag(dE_dx)
        ReE_y, ImE_y = np.real(dE_dy), np.imag(dE_dy)
        ReE_z, ImE_z = np.real(dE_dz), np.imag(dE_dz)

        # 2. Calculate N = Re(E) x Im(E)
        n_vec = np.cross(ReE, ImE)
        
        # We return 2 constraints to satisfy the codimension-2 requirement.
        vals = np.array([n_vec[1], n_vec[2]])

        # 3. Jacobian of N        
        # Partial derivatives of N vector with respect to x, y, z
        dn_dx = np.cross(ReE_x, ImE) + np.cross(ReE, ImE_x)
        dn_dy = np.cross(ReE_y, ImE) + np.cross(ReE, ImE_y)
        dn_dz = np.cross(ReE_z, ImE) + np.cross(ReE, ImE_z)
        
        # Construct 2x3 Jacobian 
        # Rows correspond to gradients of N_y and N_z
        jac = np.array([
            [dn_dx[1], dn_dy[1], dn_dz[1]],
            [dn_dx[2], dn_dy[2], dn_dz[2]],
        ])

        return vals, jac

    def trace_L_lines(self, starting_points, ds=0.05, max_steps=500, max_iter=10, value_tol=1e-6):
        """
        Traces L^T lines in 3D from seed points (Re(E) x Im(E)=0 curves).

        Args:
            starting_points: List of seed dicts from find_L_T_points() or (x,y,z) tuples
            ds: Step size along tangent to cruve
            max_steps: Maximum number of tracing steps
            max_iter: Corrector iterations per step
            value_tol: Residual tolerance for normal vector
            
        Returns:
            List of (N, 3) trajectory arrays
        """
        all_lines = []
        for seed in starting_points:
            line = self._trace_line_generic(
                seed, 
                self._L_T_val_jac_3d, ds, 
                max_steps, max_iter, value_tol
            )
            all_lines.append(line)
        return all_lines

