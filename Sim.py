import numpy as np
L = 100.0
nx = 400 
g  = 9.81  

bed_slope   = -0.08
basin_x1    = 45.0
basin_x2    = 75.0
basin_depth = 1.2

pile_center = 8.0
pile_sigma  = 4.0
pile_height = 1.0
h_floor     = 0.0 

dx = L / nx
x  = (np.arange(nx) + 0.5) * dx

plane = bed_slope * x

b = np.copy(plane)
mask = (x >= basin_x1) & (x <= basin_x2)
if np.any(mask):
    xi = (x[mask] - basin_x1) / (basin_x2 - basin_x1) 
    bowl = basin_depth * (0.5 - 0.5 * np.cos(2 * np.pi * xi))
    b[mask] = plane[mask] - bowl


h = pile_height * np.exp(-0.5 * ((x - pile_center) / pile_sigma) ** 2)
if h_floor > 0.0:
    h = np.maximum(h, h_floor)

u = np.zeros_like(h)
W = h + b 
q = h * u  

state = {
    "x": x, "dx": dx, "b": b, "h": h, "u": u, "W": W, "q": q,
    "params": {
        "L": L, "nx": nx, "g": g, "bed_slope": bed_slope,
        "basin_x1": basin_x1, "basin_x2": basin_x2, "basin_depth": basin_depth,
        "pile_center": pile_center, "pile_sigma": pile_sigma, "pile_height": pile_height,
        "h_floor": h_floor,
    },
}

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.figure(figsize=(9,4.2))
    plt.plot(x, b, label="bed b(x)")
    plt.plot(x, W, label="free surface W=h+b")
    plt.plot(x, h, label="thickness h(x)")
    plt.xlabel("x (m)"); plt.ylabel("elevation / thickness (m)")
    plt.title("Step 1: Initialized bed, W, and h (single-block script)")
    plt.legend(); plt.tight_layout(); plt.show()
