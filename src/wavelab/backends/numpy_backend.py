import numpy as np
import time
from dataclasses import dataclass

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(total=None, disable=False, desc=None, **kwargs):
        class DummyPbar:
            def update(self, n): pass
            def close(self): pass
        return DummyPbar()

def _kernel_grid_rect(x_vec, y_vec, z, t, kx, ky, kz, cx, cy, cz, w, inv_w, 
                      need_b, need_derivs, 
                      E_out, B_out, dx_out, dy_out, dz_out):
    """
    Vectorized grid kernel.
    """    
    kx_v = kx[:, None, None]
    ky_v = ky[:, None, None]
    kz_v = kz[:, None, None]
    w_v = w[:, None, None]
    
    X = x_vec[None, None, :]
    Y = y_vec[None, :, None]
    
    phase = (kx_v * X) + (ky_v * Y) + (kz_v * z) - (w_v * t)
    wf = np.exp(1j * phase)

    E_out[0] += np.sum(cx[:, None, None] * wf, axis=0)
    E_out[1] += np.sum(cy[:, None, None] * wf, axis=0)
    E_out[2] += np.sum(cz[:, None, None] * wf, axis=0)

    if need_b:
        bx_c = ((ky * cz - kz * cy) * inv_w)[:, None, None]
        by_c = ((kz * cx - kx * cz) * inv_w)[:, None, None]
        bz_c = ((kx * cy - ky * cx) * inv_w)[:, None, None]

        B_out[0] += np.sum(bx_c * wf, axis=0)
        B_out[1] += np.sum(by_c * wf, axis=0)
        B_out[2] += np.sum(bz_c * wf, axis=0)

    if need_derivs:
        ikx = (1j * kx)[:, None, None]
        iky = (1j * ky)[:, None, None]
        ikz = (1j * kz)[:, None, None]
        
        cx_v = cx[:, None, None]
        cy_v = cy[:, None, None]
        cz_v = cz[:, None, None]

        dx_out[0] += np.sum(ikx * cx_v * wf, axis=0)
        dx_out[1] += np.sum(ikx * cy_v * wf, axis=0)
        dx_out[2] += np.sum(ikx * cz_v * wf, axis=0)

        dy_out[0] += np.sum(iky * cx_v * wf, axis=0)
        dy_out[1] += np.sum(iky * cy_v * wf, axis=0)
        dy_out[2] += np.sum(iky * cz_v * wf, axis=0)

        dz_out[0] += np.sum(ikz * cx_v * wf, axis=0)
        dz_out[1] += np.sum(ikz * cy_v * wf, axis=0)
        dz_out[2] += np.sum(ikz * cz_v * wf, axis=0)

def _kernel_cloud_flat(x_arr, y_arr, z_arr, t, kx, ky, kz, cx, cy, cz, w, inv_w, 
                  need_b, need_derivs, 
                  E_out, B_out, dx_out, dy_out, dz_out):
    """
    Vectorized cloud kernel. 
    """    
    kx_v = kx[:, None]; ky_v = ky[:, None]; kz_v = kz[:, None]; w_v = w[:, None]
    
    x = x_arr[None, :]
    y = y_arr[None, :]
    z = z_arr[None, :]

    phase = (kx_v * x) + (ky_v * y) + (kz_v * z) - (w_v * t)
    wf = np.exp(1j * phase)

    E_out[0] = np.sum(cx[:, None] * wf, axis=0)
    E_out[1] = np.sum(cy[:, None] * wf, axis=0)
    E_out[2] = np.sum(cz[:, None] * wf, axis=0)

    if need_b:
        bx_c = ((ky * cz - kz * cy) * inv_w)[:, None]
        by_c = ((kz * cx - kx * cz) * inv_w)[:, None]
        bz_c = ((kx * cy - ky * cx) * inv_w)[:, None]

        B_out[0] = np.sum(bx_c * wf, axis=0)
        B_out[1] = np.sum(by_c * wf, axis=0)
        B_out[2] = np.sum(bz_c * wf, axis=0)

    if need_derivs:
        ikx = (1j * kx)[:, None]
        iky = (1j * ky)[:, None]
        ikz = (1j * kz)[:, None]
        
        cx_v = cx[:, None]; cy_v = cy[:, None]; cz_v = cz[:, None]

        dx_out[0] = np.sum(ikx * cx_v * wf, axis=0)
        dx_out[1] = np.sum(ikx * cy_v * wf, axis=0)
        dx_out[2] = np.sum(ikx * cz_v * wf, axis=0)

        dy_out[0] = np.sum(iky * cx_v * wf, axis=0)
        dy_out[1] = np.sum(iky * cy_v * wf, axis=0)
        dy_out[2] = np.sum(iky * cz_v * wf, axis=0)

        dz_out[0] = np.sum(ikz * cx_v * wf, axis=0)
        dz_out[1] = np.sum(ikz * cy_v * wf, axis=0)
        dz_out[2] = np.sum(ikz * cz_v * wf, axis=0)

def _kernel_point(x, y, z, t, kx, ky, kz, cx, cy, cz, w, inv_w, need_b, need_derivs):
    phase = (kx * x + ky * y + kz * z) - (w * t)
    wf = np.exp(1j * phase)

    ex = np.sum(cx * wf)
    ey = np.sum(cy * wf)
    ez = np.sum(cz * wf)

    E = np.array([ex, ey, ez], dtype=np.complex128)
    
    B = np.empty(3, dtype=np.complex128)
    if need_b:
        bx = np.sum(((ky * cz - kz * cy) * inv_w) * wf)
        by = np.sum(((kz * cx - kx * cz) * inv_w) * wf)
        bz = np.sum(((kx * cy - ky * cx) * inv_w) * wf)
        B[:] = [bx, by, bz]

    dx = np.empty(3, dtype=np.complex128)
    dy = np.empty(3, dtype=np.complex128)
    dz = np.empty(3, dtype=np.complex128)
    
    if need_derivs:
        ikx, iky, ikz = 1j*kx, 1j*ky, 1j*kz
        dx[:] = [np.sum(ikx*cx*wf), np.sum(ikx*cy*wf), np.sum(ikx*cz*wf)]
        dy[:] = [np.sum(iky*cx*wf), np.sum(iky*cy*wf), np.sum(iky*cz*wf)]
        dz[:] = [np.sum(ikz*cx*wf), np.sum(ikz*cy*wf), np.sum(ikz*cz*wf)]

    return E, (dx, dy, dz), B

class NumpyMethods:
    def __init__(self, beam, max_points_per_batch=4096):
        self.kx, self.ky, self.kz = beam.k
        self.cx, self.cy, self.cz = beam.c
        self.w, self.inv_w = beam.w, beam.inv_w
        self.max_batch_size = max_points_per_batch

    def _allocate_arrays(self, shape):
        return tuple(np.zeros((3,*shape), dtype=np.complex128) for _ in range(5))
        
    def compute_cloud(self, x, y, z, t, need_b=True, need_derivs=True, progress_bar=False):
        total_points = len(x)
        # Dynamic batch sizing to prevent OOM
        batch_size = self.max_batch_size
        
        E, B, dx, dy, dz = self._allocate_arrays((total_points,))
        pbar = tqdm(total=total_points, disable=not progress_bar, desc="Cloud (NP)", unit="pts")
        
        for i in range(0, total_points, batch_size):
            end = min(i + batch_size, total_points)
            _kernel_cloud_flat(
                x[i:end], y[i:end], z[i:end], t, 
                self.kx, self.ky, self.kz, self.cx, self.cy, self.cz, self.w, self.inv_w, 
                need_b, need_derivs, 
                E[:][:, i:end], B[:][:, i:end], dx[:][:, i:end], dy[:][:, i:end], dz[:][:, i:end]
            )
            pbar.update(end - i)
            
        pbar.close()
        D = (dx, dy, dz) if need_derivs else (None, None, None)
        B = B if need_b else None

        return E, D, B

    def compute_grid(self, x_vec, y_vec, z, t, need_b=True, need_derivs=True, progress_bar=False):
        nx, ny = len(x_vec), len(y_vec)
        rows_per_batch = max(1, self.max_batch_size // nx)
        
        E, B, dx, dy, dz = self._allocate_arrays((ny, nx))
        pbar = tqdm(total=ny, disable=not progress_bar, desc="Grid (NP)", unit="rows")
        
        for i in range(0, ny, rows_per_batch):
            end = min(i + rows_per_batch, ny)
            _kernel_grid_rect(
                x_vec, y_vec[i:end], z, t, 
                self.kx, self.ky, self.kz, self.cx, self.cy, self.cz, self.w, self.inv_w, 
                need_b, need_derivs, 
                E[:][:, i:end, :], B[:][:, i:end, :], dx[:][:, i:end, :], dy[:][:, i:end, :], dz[:][:, i:end, :]
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
    
    interface = NumpyMethods(beam)
    
    x_vec, y_vec = np.linspace(-5, 5, grid_size), np.linspace(-5, 5, grid_size)
    X, Y = np.meshgrid(x_vec, y_vec)
    x_arr, y_arr, z_arr = X.ravel(), Y.ravel(), np.zeros_like(X.ravel())
    
    print(f"NumPy Implementation | Waves: {num_waves} | Points: {len(x_arr):,}")
    
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