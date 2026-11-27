#!/usr/bin/env python3
"""
Mobility and structure descriptors of the step-2 aggregate classes (step2_generate.py output).

Mobility is computed by an external module whose path is given on the command line.  The
module must provide
  capacitance(pos, rad, rng=...)  -> Smoluchowski radius R_S [um] of the union of spheres
                                     (pos shape (N, 3) [um], rad shape (N,) [um])
  mobility_diameter(r_s, pa)      -> (d_mob [um], Kn) for R_S [um] and projected area PA [um^2]
as in the adjusted-sphere method of Zhang et al. (2012, Aerosol Sci. Technol. 46, 1065) after
Dahneke (1973).  The results in this directory used a walk-on-spheres capacitance (20000 walks)
and air at 296.15 K and 101.3 kPa (mean free path 0.0673 um; slip correction constants 1.257,
0.400, 1.100).
  PA    orientation-averaged projected area, computed here (dem_shape.projected_area, 64
        directions)
Lengths: a = A_UM micrometres.

Structure: Rg_F (self term a**2), coordination numbers for surface gaps below 0.02 a and
0.05 a (PTSA leaves gaps up to about 0.04 a by design), convexity N (4/3) pi a**3 / V_hull of the
centres, and the gyration-tensor ratios sqrt(l1/l3) and l1/l2.

Usage:
    python step2_mobility.py step2.h5 MOBILITY_MODULE.py out.csv [n_proc]
"""
import importlib.util
import json
import os
import sys
import zlib
from multiprocessing import Pool

import h5py
import numpy as np
from scipy.spatial import ConvexHull, cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dem_shape  # noqa: E402

A_UM = 0.015  # monomer radius [um]
DIRS = dem_shape.fibonacci_directions(64)
MA = None


def init(path):
    global MA
    spec = importlib.util.spec_from_file_location("mobility_model", path)
    MA = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(MA)


def one(args):
    name, i, xp = args  # xp shape: (N, 3), units of a
    n = len(xp)
    seed = zlib.crc32(f"{name}/{i}".encode())  # reproducible across processes
    r_s = MA.capacitance(xp * A_UM, np.full(n, A_UM), rng=np.random.default_rng(seed))
    pa = dem_shape.projected_area(xp, DIRS) * A_UM ** 2
    d_mob, kn = MA.mobility_diameter(r_s, pa)
    xc = xp - xp.mean(0)
    eig = np.sort(np.linalg.eigvalsh(xc.T @ xc / n))[::-1]
    tree = cKDTree(xp)
    coord = [2.0 * len(tree.query_pairs(2.0 + g)) / n for g in (0.02, 0.05)]
    conv = n * 4.0 / 3.0 * np.pi / ConvexHull(xp).volume
    rgf = np.sqrt((xc ** 2).sum(1).mean() + 1.0)
    return (f"{name},{i},{rgf:.6f},{r_s:.8e},{pa:.8e},{d_mob:.8e},{kn:.6f},"
            f"{coord[0]:.5f},{coord[1]:.5f},{conv:.6f},{np.sqrt(eig[0] / eig[2]):.5f},"
            f"{eig[0] / eig[1]:.5f}")


def main(h5path, ma_path, out, n_proc=20):
    jobs = []
    with h5py.File(h5path, "r") as h:
        for name in json.loads(h.attrs["classes"]):
            xs = h[name]["xp"][:]  # shape: (n, N, 3)
            jobs += [(name, i, x) for i, x in enumerate(xs)]
    with Pool(int(n_proc), initializer=init, initargs=(ma_path,)) as pool:
        rows = pool.map(one, jobs, chunksize=4)
    with open(out, "w") as f:
        f.write("class,idx,rgf,R_S_um,PA_um2,d_mob_um,Kn,coord_002,coord_005,conv3,"
                "ax_l1l3,ax_l1l2\n")
        f.write("\n".join(rows) + "\n")
    print(f"saved {out}: {len(rows)} aggregates")


if __name__ == "__main__":
    main(*sys.argv[1:])
