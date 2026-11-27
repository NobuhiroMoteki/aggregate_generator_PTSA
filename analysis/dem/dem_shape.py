#!/usr/bin/env python3
"""
Shape descriptors of soot-dem restructuring runs for optical and mobility calculations.
The Smoluchowski radius needed for the continuum-regime mobility is not computed here;
it is left to the capacitance code of the mobility project, using the saved coordinates.

Reads run/particles_<k>.vtk of every run directory listed in WORK_DIR/jobs.txt.
Positions in the dumps are in units of the monomer radius a (all monomers have radius a).

Per run and dump (all dumps):
  rgf     : radius of gyration with self term, Rg_F**2 = <|x - x_c|**2> + a**2 [a]
            (the convention of the tcca placement law, N = k_f (Rg_F/a)**Df)
For the first and the last dump:
  eig     : eigenvalues of the gyration tensor of the centres, descending [a**2], shape (2, 3)
  ap      : orientation-averaged projected area of the union of the spheres [a**2]
            (n_dir quasi-uniform directions, rasterised with pixel PIX)
  sq      : Debye structure factor of point scatterers at the centres,
            S(q) = 1 + (2/N) sum_{i<j} sin(q r_ij)/(q r_ij), on the grid QA [1], shape (2, nq)
  pairs   : counts of pairs i < j per centre-distance bin EDGES [1], shape (2, nb)
  xp0, xp1: centre coordinates of the first and the last dump [a], shape (N, 3)

Usage (on the host that holds the work directories):
    python dem_shape.py WORK_DIR out.npz [n_proc] [n_dir]
"""
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np

PIX = 0.05                              # raster pixel [a]
QA = np.logspace(-2, np.log10(3.0), 60)  # q a grid, shape: (60,)
EDGES = np.logspace(np.log10(1.5), np.log10(80.0), 41)  # centre-distance bins [a], shape: (41,)


def read_points(path):
    """Centres from a soot-dem VTK dump, shape (N, 3), in units of a."""
    with open(path) as f:
        lines = f.read().split("\n")
    i = next(k for k, ln in enumerate(lines) if ln.startswith("POINTS"))
    n = int(lines[i].split()[1])
    vals = np.array(lines[i + 1].split(), dtype=float)  # soot-dem writes all points on one line
    return vals[:3 * n].reshape(n, 3)


def fibonacci_directions(n):
    """n quasi-uniform unit vectors on the sphere, shape (n, 3)."""
    k = np.arange(n) + 0.5
    z = 1.0 - 2.0 * k / n
    phi = np.pi * (1.0 + 5.0 ** 0.5) * k
    s = np.sqrt(1.0 - z ** 2)
    return np.stack([s * np.cos(phi), s * np.sin(phi), z], axis=1)


def union_area_2d(p, r=1.0, h=PIX):
    """Area of the union of discs of radius r centred at p (shape (N, 2)) by rasterisation."""
    m = int(np.ceil(r / h)) + 1          # stencil half-width [pixels]
    lo = p.min(0) - (m + 2.37) * h       # grid origin (off-lattice shift); the stencil stays inside
    nx, ny = (np.ceil((p.max(0) - p.min(0)) / h)).astype(int) + 2 * m + 6
    grid = np.zeros((nx, ny), dtype=bool)
    for c in p:
        i0, j0 = np.floor((c - lo) / h).astype(int)
        xs = lo[0] + (i0 + 0.5) * h + (np.arange(-m, m + 1)) * h - c[0]
        ys = lo[1] + (j0 + 0.5) * h + (np.arange(-m, m + 1)) * h - c[1]
        mask = xs[:, None] ** 2 + ys[None, :] ** 2 <= r * r
        ia, ja = i0 - m, j0 - m
        grid[ia:ia + 2 * m + 1, ja:ja + 2 * m + 1] |= mask
    return grid.sum() * h * h


def projected_area(xp, dirs):
    """Mean projected area of unit spheres at xp over the directions dirs, shape (n_dir, 3)."""
    out = np.empty(len(dirs))
    for k, d in enumerate(dirs):
        t = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(d, t)
        u /= np.linalg.norm(u)
        v = np.cross(d, u)
        out[k] = union_area_2d(np.stack([xp @ u, xp @ v], axis=1))
    return out.mean()


def descriptors(xp, dirs):
    n = len(xp)
    xc = xp - xp.mean(0)
    eig = np.sort(np.linalg.eigvalsh(xc.T @ xc / n))[::-1]
    d = np.sqrt(((xp[:, None, :] - xp[None, :, :]) ** 2).sum(-1))  # shape: (N, N)
    iu = np.triu_indices(n, 1)
    r = d[iu]  # shape: (N(N-1)/2,)
    qr = QA[:, None] * r[None, :]
    sq = 1.0 + 2.0 / n * np.sum(np.sin(qr) / qr, axis=1)
    pairs = np.histogram(r, bins=EDGES)[0]
    return eig, projected_area(xp, dirs), sq, pairs


def one_run(args):
    work, job, n_dir = args
    files = glob.glob(os.path.join(work, job, "run", "particles_*.vtk"))
    ks = sorted(int(os.path.basename(f)[10:-4]) for f in files)
    if len(ks) < 50:
        return None
    rgf = []
    for k in ks:
        x = read_points(os.path.join(work, job, "run", f"particles_{k}.vtk"))
        rgf.append(np.sqrt(((x - x.mean(0)) ** 2).sum(1).mean() + 1.0))
    x0 = read_points(os.path.join(work, job, "run", f"particles_{ks[0]}.vtk"))
    x1 = read_points(os.path.join(work, job, "run", f"particles_{ks[-1]}.vtk"))
    dirs = fibonacci_directions(n_dir)
    d0, d1 = descriptors(x0, dirs), descriptors(x1, dirs)
    agg = int(job.split("aggregate_")[1].split("/")[0])
    frac = int(job.rstrip("/").split("frac_")[1])
    return dict(agg=agg, frac=frac, rgf=np.array(rgf), xp0=x0, xp1=x1,
                eig=np.array([d0[0], d1[0]]), ap=np.array([d0[1], d1[1]]),
                sq=np.array([d0[2], d1[2]]), pairs=np.array([d0[3], d1[3]]))


def main(work, out, n_proc=10, n_dir=64):
    jobs = [j for j in open(os.path.join(work, "jobs.txt")).read().split() if j]
    with Pool(int(n_proc)) as pool:
        rows = [r for r in pool.map(one_run, [(work, j, int(n_dir)) for j in jobs]) if r]
    rows.sort(key=lambda r: (r["agg"], r["frac"]))
    np.savez_compressed(out, qa=QA, edges=EDGES, pix=PIX, n_dir=int(n_dir),
                        **{k: np.array([r[k] for r in rows]) for k in rows[0]})
    print(f"saved {out}: {len(rows)} runs")


if __name__ == "__main__":
    main(*sys.argv[1:])
