import numpy as np
import time; from dataclasses import dataclass

try: 
    from numba import jit, prange
    has_numba = True
except ImportError:
    def jit(*args, **kwargs): 
        def decor(f): return f
        return decor
    prange = range
    has_numba = False

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(total=None, disable=False, desc=None, **kwargs):
        class DummyPbar:
            def update(self, n): pass
            def close(self): pass
        return DummyPbar()

K_ARGS = {'nopython': True, 'parallel': True, 'fastmath': True, 'cache': True}

@jit(**K_ARGS)
def _kernel_grid_rect(x_vec, y_vec, z, t, kx, ky, kz, cx, cy, cz, w, inv_w, 
                      need_b, need_derivs, 
                      E_out, B_out, dx_out, dy_out, dz_out):
    nx, ny = len(x_vec), len(y_vec)
    num_waves = len(w)

    for iy in prange(ny):
        y = y_vec[iy]
        for ix in range(nx):
            x = x_vec[ix]
            
            ex, ey, ez = 0j, 0j, 0j
            bx, by, bz = 0j, 0j, 0j
            dxx, dxy, dxz, dyx, dyy, dyz, dzx, dzy, dzz = 0j,0j,0j,0j,0j,0j,0j,0j,0j

            if not need_b and not need_derivs:
                for i in range(num_waves):
                    wf = np.exp(1j * (kx[i]*x + ky[i]*y + kz[i]*z - w[i]*t))
                    ex += cx[i] * wf; ey += cy[i] * wf; ez += cz[i] * wf

            elif need_b and not need_derivs:
                for i in range(num_waves):
                    wf = np.exp(1j * (kx[i]*x + ky[i]*y + kz[i]*z - w[i]*t))
                    c_x, c_y, c_z = cx[i]*wf, cy[i]*wf, cz[i]*wf
                    ex += c_x; ey += c_y; ez += c_z

                    bx += (ky[i]*c_z - kz[i]*c_y) * inv_w[i]
                    by += (kz[i]*c_x - kx[i]*c_z) * inv_w[i]
                    bz += (kx[i]*c_y - ky[i]*c_x) * inv_w[i]

            elif not need_b and need_derivs:
                for i in range(num_waves):
                    wf = np.exp(1j * (kx[i]*x + ky[i]*y + kz[i]*z - w[i]*t))
                    c_x, c_y, c_z = cx[i]*wf, cy[i]*wf, cz[i]*wf
                    ex += c_x; ey += c_y; ez += c_z

                    ikx, iky, ikz = 1j*kx[i], 1j*ky[i], 1j*kz[i]
                    dxx += ikx*c_x; dxy += ikx*c_y; dxz += ikx*c_z
                    dyx += iky*c_x; dyy += iky*c_y; dyz += iky*c_z
                    dzx += ikz*c_x; dzy += ikz*c_y; dzz += ikz*c_z

            else: # need_b AND need_derivs
                for i in range(num_waves):
                    wf = np.exp(1j * (kx[i]*x + ky[i]*y + kz[i]*z - w[i]*t))
                    c_x, c_y, c_z = cx[i]*wf, cy[i]*wf, cz[i]*wf
                    ex += c_x; ey += c_y; ez += c_z

                    bx += (ky[i]*c_z - kz[i]*c_y) * inv_w[i]
                    by += (kz[i]*c_x - kx[i]*c_z) * inv_w[i]
                    bz += (kx[i]*c_y - ky[i]*c_x) * inv_w[i]

                    ikx, iky, ikz = 1j*kx[i], 1j*ky[i], 1j*kz[i]
                    dxx += ikx*c_x; dxy += ikx*c_y; dxz += ikx*c_z
                    dyx += iky*c_x; dyy += iky*c_y; dyz += iky*c_z
                    dzx += ikz*c_x; dzy += ikz*c_y; dzz += ikz*c_z

            E_out[0, iy, ix], E_out[1, iy, ix], E_out[2, iy, ix] = ex, ey, ez

            if need_b:
                B_out[0, iy, ix], B_out[1, iy, ix], B_out[2, iy, ix] = bx, by, bz

            if need_derivs:
                dx_out[0, iy, ix], dx_out[1, iy, ix], dx_out[2, iy, ix] = dxx, dxy, dxz
                dy_out[0, iy, ix], dy_out[1, iy, ix], dy_out[2, iy, ix] = dyx, dyy, dyz
                dz_out[0, iy, ix], dz_out[1, iy, ix], dz_out[2, iy, ix] = dzx, dzy, dzz

@jit(**K_ARGS)
def _kernel_cloud_flat(x_arr, y_arr, z_arr, t, kx, ky, kz, cx, cy, cz, w, inv_w, 
                  need_b, need_derivs, 
                  E_out, B_out, dx_out, dy_out, dz_out):
    num_points = len(x_arr)
    num_waves = len(w)

    for p in prange(num_points):
        x, y, z = x_arr[p], y_arr[p], z_arr[p]
        ex, ey, ez = 0j, 0j, 0j
        bx, by, bz = 0j, 0j, 0j
        dxx, dxy, dxz, dyx, dyy, dyz, dzx, dzy, dzz = 0j,0j,0j,0j,0j,0j,0j,0j,0j

        if not need_b and not need_derivs:
            for i in range(num_waves):
                wf = np.exp(1j * (kx[i]*x + ky[i]*y + kz[i]*z - w[i]*t))
                ex += cx[i] * wf; ey += cy[i] * wf; ez += cz[i] * wf

        elif need_b and not need_derivs:
            for i in range(num_waves):
                wf = np.exp(1j * (kx[i]*x + ky[i]*y + kz[i]*z - w[i]*t))
                c_x, c_y, c_z = cx[i]*wf, cy[i]*wf, cz[i]*wf
                ex += c_x; ey += c_y; ez += c_z

                bx += (ky[i]*c_z - kz[i]*c_y)*inv_w[i]
                by += (kz[i]*c_x - kx[i]*c_z)*inv_w[i]
                bz += (kx[i]*c_y - ky[i]*c_x)*inv_w[i]

        elif not need_b and need_derivs:
            for i in range(num_waves):
                wf = np.exp(1j * (kx[i]*x + ky[i]*y + kz[i]*z - w[i]*t))
                c_x, c_y, c_z = cx[i]*wf, cy[i]*wf, cz[i]*wf
                ex += c_x; ey += c_y; ez += c_z
                
                ikx, iky, ikz = 1j*kx[i], 1j*ky[i], 1j*kz[i]
                dxx += ikx*c_x; dxy += ikx*c_y; dxz += ikx*c_z
                dyx += iky*c_x; dyy += iky*c_y; dyz += iky*c_z
                dzx += ikz*c_x; dzy += ikz*c_y; dzz += ikz*c_z

        else:
            for i in range(num_waves):
                wf = np.exp(1j * (kx[i]*x + ky[i]*y + kz[i]*z - w[i]*t))
                c_x, c_y, c_z = cx[i]*wf, cy[i]*wf, cz[i]*wf
                ex += c_x; ey += c_y; ez += c_z

                bx += (ky[i]*c_z - kz[i]*c_y)*inv_w[i]
                by += (kz[i]*c_x - kx[i]*c_z)*inv_w[i]
                bz += (kx[i]*c_y - ky[i]*c_x)*inv_w[i]

                ikx, iky, ikz = 1j*kx[i], 1j*ky[i], 1j*kz[i]
                dxx += ikx*c_x; dxy += ikx*c_y; dxz += ikx*c_z
                dyx += iky*c_x; dyy += iky*c_y; dyz += iky*c_z
                dzx += ikz*c_x; dzy += ikz*c_y; dzz += ikz*c_z

        E_out[0, p], E_out[1, p], E_out[2, p] = ex, ey, ez

        if need_b: B_out[0, p], B_out[1, p], B_out[2, p] = bx, by, bz

        if need_derivs:
            dx_out[0,p], dx_out[1,p], dx_out[2,p] = dxx, dxy, dxz
            dy_out[0,p], dy_out[1,p], dy_out[2,p] = dyx, dyy, dyz
            dz_out[0,p], dz_out[1,p], dz_out[2,p] = dzx, dzy, dzz

@jit(nopython=True, parallel=False, fastmath=True, cache=True)
def _kernel_point(x, y, z, t, kx, ky, kz, cx, cy, cz, w, inv_w, need_b, need_derivs):
    num_waves = len(w)
    ex, ey, ez = 0j, 0j, 0j
    bx, by, bz = 0j, 0j, 0j
    dxx, dxy, dxz, dyx, dyy, dyz, dzx, dzy, dzz = 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j

    if not need_b and not need_derivs:
        for i in range(num_waves):
            wf = np.exp(1j * (kx[i]*x + ky[i]*y + kz[i]*z - w[i]*t))
            ex += cx[i] * wf; ey += cy[i] * wf; ez += cz[i] * wf

    elif need_b and not need_derivs:
        for i in range(num_waves):
            wf = np.exp(1j * (kx[i]*x + ky[i]*y + kz[i]*z - w[i]*t))
            c_x, c_y, c_z = cx[i]*wf, cy[i]*wf, cz[i]*wf
            ex += c_x; ey += c_y; ez += c_z

            bx += (ky[i]*c_z - kz[i]*c_y)*inv_w[i]
            by += (kz[i]*c_x - kx[i]*c_z)*inv_w[i]
            bz += (kx[i]*c_y - ky[i]*c_x)*inv_w[i]

    elif not need_b and need_derivs:
        for i in range(num_waves):
            wf = np.exp(1j * (kx[i]*x + ky[i]*y + kz[i]*z - w[i]*t))
            c_x, c_y, c_z = cx[i]*wf, cy[i]*wf, cz[i]*wf
            ex += c_x; ey += c_y; ez += c_z

            ikx, iky, ikz = 1j*kx[i], 1j*ky[i], 1j*kz[i]
            dxx += ikx*c_x; dxy += ikx*c_y; dxz += ikx*c_z
            dyx += iky*c_x; dyy += iky*c_y; dyz += iky*c_z
            dzx += ikz*c_x; dzy += ikz*c_y; dzz += ikz*c_z

    else:
        for i in range(num_waves):
            wf = np.exp(1j * (kx[i]*x + ky[i]*y + kz[i]*z - w[i]*t))
            c_x, c_y, c_z = cx[i]*wf, cy[i]*wf, cz[i]*wf
            ex += c_x; ey += c_y; ez += c_z

            bx += (ky[i]*c_z - kz[i]*c_y)*inv_w[i]
            by += (kz[i]*c_x - kx[i]*c_z)*inv_w[i]
            bz += (kx[i]*c_y - ky[i]*c_x)*inv_w[i]

            ikx, iky, ikz = 1j*kx[i], 1j*ky[i], 1j*kz[i]
            dxx += ikx*c_x; dxy += ikx*c_y; dxz += ikx*c_z
            dyx += iky*c_x; dyy += iky*c_y; dyz += iky*c_z
            dzx += ikz*c_x; dzy += ikz*c_y; dzz += ikz*c_z

    E = np.empty(3, dtype=np.complex128)
    E[0], E[1], E[2] = ex, ey, ez

    B = np.empty(3, dtype=np.complex128)
    if need_b: B[0], B[1], B[2] = bx, by, bz

    dx = np.empty(3, dtype=np.complex128)
    dy = np.empty(3, dtype=np.complex128)
    dz = np.empty(3, dtype=np.complex128)
    if need_derivs:
        dx[0], dx[1], dx[2] = dxx, dxy, dxz
        dy[0], dy[1], dy[2] = dyx, dyy, dyz
        dz[0], dz[1], dz[2] = dzx, dzy, dzz

    return E, (dx, dy, dz), B

class NumbaMethods:
    def __init__(self, beam, max_points_per_batch=250_000):
        self.kx, self.ky, self.kz = beam.k
        self.cx, self.cy, self.cz = beam.c
        self.w, self.inv_w = beam.w, beam.inv_w
        self.max_batch_size = max_points_per_batch

    def _allocate_arrays(self, shape):
        return tuple(np.empty((3,*shape), dtype=np.complex128) for _ in range(5))
        
    def compute_cloud(self, x, y, z, t, need_b=True, need_derivs=True, progress_bar=False):
        total_points = len(x)
        batch_size = self.max_batch_size
        if progress_bar: 
            batch_size = min(self.max_batch_size, max(1, total_points // 5))  

        E, B, dx, dy, dz = self._allocate_arrays((total_points,))
        pbar = tqdm(total=total_points, disable=not progress_bar, desc="Cloud", unit="pts")
        
        for i in range(0, total_points, batch_size):
            end = min(i + batch_size, total_points)
            _kernel_cloud_flat(
                x[i:end], y[i:end], z[i:end], t, 
                self.kx, self.ky, self.kz, self.cx, self.cy, self.cz, self.w, self.inv_w, 
                need_b, need_derivs, 
                E[:, i:end], B[:, i:end], dx[:,i:end], dy[:, i:end], dz[:, i:end]
            )
            pbar.update(end - i)
            
        pbar.close()
        D = (dx, dy, dz) if need_derivs else (None, None, None)
        B = B if need_b else None

        return E, D, B

    def compute_grid(self, x_vec, y_vec, z, t, need_b=True, need_derivs=True, progress_bar=False):
        nx, ny = len(x_vec), len(y_vec)
        rows_per_batch = self.max_batch_size
        if progress_bar:
            rows_per_batch = min(self.max_batch_size, max(1, ny // 5))
            
        E, B, dx, dy, dz = self._allocate_arrays((ny, nx))
        pbar = tqdm(total=ny, disable=not progress_bar, desc="Grid", unit="rows")
        
        for i in range(0, ny, rows_per_batch):
            end = min(i + rows_per_batch, ny)
            _kernel_grid_rect(
                x_vec, y_vec[i:end], z, t, 
                self.kx, self.ky, self.kz, self.cx, self.cy, self.cz, self.w, self.inv_w, 
                need_b, need_derivs, 
                E[:, i:end, :], B[:, i:end, :], dx[:, i:end, :], dy[:, i:end, :], dz[:, i:end, :]
            )
            pbar.update(end - i)
            
        pbar.close()
        D = (dx, dy, dz) if need_derivs else (None, None, None)
        B = B if need_b else None

        return E, D, B
    
    def compute_point(self, x, y, z, t, need_b=True, need_derivs=True):
        E, D, B = _kernel_point(
            x, y, z, t, self.kx, self.ky, self.kz, self.cx, self.cy, self.cz, 
            self.w, self.inv_w, need_b, need_derivs
        )
        if not need_derivs: D = (None, None, None)
        if not need_b: B = None
        return E, D, B

def run_benchmark():
    num_waves = 5000

    @dataclass
    class beam:
        k = np.random.random((3, num_waves))
        c = np.random.random((3, num_waves))
        w = np.linalg.norm(k, axis=0)
        inv_w = 1.0 / w

    grid_size = 100 
    
    interface = NumbaMethods(beam)
    
    x_vec, y_vec = np.linspace(-5, 5, grid_size), np.linspace(-5, 5, grid_size)
    X, Y = np.meshgrid(x_vec, y_vec)
    x_arr, y_arr, z_arr = X.ravel(), Y.ravel(), np.zeros_like(X.ravel())
    
    print(f"Numba Implementation | Waves: {num_waves} | Points: {len(x_arr):,}")
    
    # Warmups
    _, _, _ = interface.compute_grid(x_vec[:2], y_vec[:2], 0.0, 0.0)
    _, _, _ = interface.compute_cloud(x_arr[:4], y_arr[:4], z_arr[:4], 0.0)
    _, _, _ = interface.compute_point(1,1,1,1)
    
    print("\nStarting Grid benchmark...")
    t0 = time.perf_counter()
    interface.compute_grid(x_vec, y_vec, 0.0, 0.0, progress_bar=True)
    print(f"Grid time: {time.perf_counter() - t0:.4f}s")
    
    print("\nStarting Cloud benchmark...")
    t0 = time.perf_counter()
    interface.compute_cloud(x_arr, y_arr, z_arr, 0.0, progress_bar=True)
    print(f"Cloud time: {time.perf_counter() - t0:.4f}s")

    print("\nStarting Point benchmark...")
    t0 = time.perf_counter()
    interface.compute_point(1,1,1,1)
    print(f"point time: {time.perf_counter() - t0:.4f}s")


if __name__ == "__main__":
    run_benchmark()