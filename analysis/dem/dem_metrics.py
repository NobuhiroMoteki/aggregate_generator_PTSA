#!/usr/bin/env python3
"""
Morphology metrics of soot-dem restructuring runs (reproduction of Demidov et al. 2024, Fig. 8).

Reads run/particles_<k>.vtk (positions in units of the monomer radius r) and computes, per dump:
  coord   : mean coordination number; neighbours have surface gap < 1 % of the diameter (0.02 r)
  conv3_s : 3D convexity = N*(4/3)*pi / V_hull, hull of points sampled on the sphere surfaces
  conv3_c : same with the hull of the sphere centres
  conv2   : 2D convexity, averaged over the xy, xz and yz projections (area by rasterisation,
            pixel 0.05 r), as a pair (projected area / hull of the projected discs,
            projected area / hull of the projected centres); stored for the first (conv2_0)
            and the last dump (conv2_1)
Demidov et al. (2024) use the hull of the centres for both the 3D and the 2D convexity
(conv3_c and the second element of conv2); the other variants are kept for comparison.

Usage (from the repository root):
    python analysis/dem/dem_metrics.py WORK_DIR out.npz
WORK_DIR is a work directory prepared by prepare_inputs.py after the runs have finished.
"""
import glob
import os
import sys

import numpy as np
from scipy.spatial import ConvexHull, cKDTree


def read_points(path):
    with open(path) as f:
        lines = f.read().split("\n")
    i = next(k for k, l in enumerate(lines) if l.startswith("POINTS"))
    n = int(lines[i].split()[1])
    vals = np.array(lines[i + 1].split(), dtype=float)
    return vals[:3 * n].reshape(n, 3)  # shape: (n, 3), units of r


def fib_sphere(m):
    k = np.arange(m) + 0.5
    phi = np.arccos(1 - 2 * k / m)
    th = np.pi * (1 + 5 ** 0.5) * k
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


SPH = fib_sphere(200)  # shape: (200, 3)
CIRC = np.stack([np.cos(np.linspace(0, 2 * np.pi, 72, endpoint=False)),
                 np.sin(np.linspace(0, 2 * np.pi, 72, endpoint=False))], 1)  # shape: (72, 2)


def coordination(x):
    pairs = cKDTree(x).query_pairs(2.02)
    return 2 * len(pairs) / len(x)


def convexity3(x):
    v = len(x) * 4 / 3 * np.pi
    pts = (x[:, None, :] + SPH[None]).reshape(-1, 3)
    return v / ConvexHull(pts).volume, v / ConvexHull(x).volume


def convexity2(x, h=0.05):
    out = []
    for a, b in ((0, 1), (0, 2), (1, 2)):
        p = x[:, [a, b]]
        lo = p.min(0) - 1.0
        hi = p.max(0) + 1.0
        gx = np.arange(lo[0], hi[0], h) + h / 2
        gy = np.arange(lo[1], hi[1], h) + h / 2
        X, Y = np.meshgrid(gx, gy, indexing="ij")
        covered = np.zeros(X.shape, bool)
        for c in p:
            i0, i1 = np.searchsorted(gx, [c[0] - 1, c[0] + 1])
            j0, j1 = np.searchsorted(gy, [c[1] - 1, c[1] + 1])
            sub = (X[i0:i1, j0:j1] - c[0]) ** 2 + (Y[i0:i1, j0:j1] - c[1]) ** 2 <= 1.0
            covered[i0:i1, j0:j1] |= sub
        area = covered.sum() * h * h
        hull = ConvexHull((p[:, None, :] + CIRC[None]).reshape(-1, 2)).volume
        out.append((area / hull, area / ConvexHull(p).volume))
    return tuple(np.mean(np.array(out), axis=0))  # (hull of discs, hull of centres)


def main(root, out):
    rows = []
    for d in sorted(glob.glob(os.path.join(root, "restructuring_simulations/size_150/aggregate_*/thin_coating/frac_*"))):
        files = glob.glob(os.path.join(d, "run", "particles_*.vtk"))
        if len(files) < 50:
            continue
        agg = int(d.split("aggregate_")[1].split("/")[0])
        frac = int(d.rstrip("/").split("frac_")[1])
        ks = sorted(int(os.path.basename(f)[10:-4]) for f in files)
        co, c3s, c3c = [], [], []
        for k in ks:
            x = read_points(os.path.join(d, "run", f"particles_{k}.vtk"))
            co.append(coordination(x))
            a, b = convexity3(x)
            c3s.append(a)
            c3c.append(b)
        x0 = read_points(os.path.join(d, "run", f"particles_{ks[0]}.vtk"))
        x1 = read_points(os.path.join(d, "run", f"particles_{ks[-1]}.vtk"))
        rows.append(dict(agg=agg, frac=frac, coord=co, conv3_s=c3s, conv3_c=c3c,
                         conv2_0=convexity2(x0), conv2_1=convexity2(x1)))
        print(agg, frac, f"coord {co[0]:.2f}->{co[-1]:.2f}  conv3_s {c3s[0]:.3f}->{c3s[-1]:.3f}  "
              f"conv3_c {c3c[0]:.3f}->{c3c[-1]:.3f}  conv2 {np.round(rows[-1]['conv2_0'], 3)}->{np.round(rows[-1]['conv2_1'], 3)}",
              flush=True)
    np.savez(out, agg=np.array([r["agg"] for r in rows]), frac=np.array([r["frac"] for r in rows]),
             coord=np.array([r["coord"] for r in rows]), conv3_s=np.array([r["conv3_s"] for r in rows]),
             conv3_c=np.array([r["conv3_c"] for r in rows]),
             conv2_0=np.array([r["conv2_0"] for r in rows]), conv2_1=np.array([r["conv2_1"] for r in rows]))
    print(f"saved {out}: {len(rows)} runs")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
