#!/usr/bin/env python3
"""
Check of the ellipsoid estimate of f_A/f_H against a projected-area tensor estimate.

f_A / f_H is the ratio of the arithmetic to the harmonic orientational mean of the free
molecular friction tensor K: f_A = tr(K)/3, f_H = 3/tr(K^-1).  Two estimates are compared:
  ellipsoid  : convex-body tensor of the ellipsoid with the same gyration tensor (semi-axes
               sqrt(5) g_k), from ANISOTROPY_MODULE.friction_ratio_ellipsoid
  PA tensor  : quadratic form n^T K n fitted to the projected area PA(n) of the aggregate in
               N_DIR directions; it includes the shielding between monomers but assumes that the
               directional friction is proportional to PA(n)
The PA-tensor method is also applied to the equivalent ellipsoid itself (analytic PA(n)), which
shows its bias relative to the exact convex-body tensor.  Neither estimate includes repeated
collisions or the coupling of translation and rotation.

Usage (from the repository root):
    python analysis/dem/step2_pa_tensor.py ANISOTROPY_MODULE.py [n_per_class]
"""
import importlib.util
import os
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dem_shape  # noqa: E402

N_DIR = 128
CLASSES = {"step2_aggregates.h5": ["I/flage", "R/flage/0", "R/flage/100", "G/ptsa/2.0/0",
                                   "G/tcca/2.0/0"],
           "step2b_aggregates.h5": ["M/ptsa/2.6/7.0", "M/ptsa/2.6/10.0"]}


def quad_basis(dirs):
    x, y, z = dirs.T
    return np.stack([x * x, y * y, z * z, 2 * x * y, 2 * x * z, 2 * y * z], axis=1)


def tensor_ratio(basis, values):
    """f_A/f_H of the symmetric tensor whose quadratic form fits values; and the rms misfit."""
    c, *_ = np.linalg.lstsq(basis, values, rcond=None)
    K = np.array([[c[0], c[3], c[4]], [c[3], c[1], c[5]], [c[4], c[5], c[2]]])
    e = np.linalg.eigvalsh(K)
    return e.mean() * np.mean(1.0 / e), np.sqrt(np.mean((basis @ c - values) ** 2)) / values.mean()


def pa_directional(xp, dirs):
    out = np.empty(len(dirs))
    for k, d in enumerate(dirs):
        t = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(d, t)
        u /= np.linalg.norm(u)
        v = np.cross(d, u)
        out[k] = dem_shape.union_area_2d(np.stack([xp @ u, xp @ v], axis=1))
    return out


def main(aniso_path, n=12):
    spec = importlib.util.spec_from_file_location("anisotropy_model", aniso_path)
    an = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(an)
    dirs = dem_shape.fibonacci_directions(N_DIR)
    basis = quad_basis(dirs)
    dirs_e = dem_shape.fibonacci_directions(2000)
    basis_e = quad_basis(dirs_e)
    print("f_A/f_H - 1 [%], mean over aggregates (units of a, monodisperse)")
    print(f"{'class':16s} {'PA tensor':>10s} {'ellipsoid':>10s} {'PA tensor on':>13s} "
          f"{'PAmax/PAmin':>12s} {'fit rms':>8s}")
    print(f"{'':16s} {'aggregate':>10s} {'(convex K)':>10s} {'ellipsoid':>13s}")
    for fname, classes in CLASSES.items():
        with h5py.File(os.path.join(HERE, fname), "r") as h:
            for c in classes:
                rows = []
                for x in h[c]["xp"][:int(n)]:
                    p = pa_directional(x, dirs)
                    r_pa, rms = tensor_ratio(basis, p)
                    a = np.sqrt(5.0) * an.principal_radii(x, np.ones(len(x)))
                    r_ell = an.friction_ratio_ellipsoid(a)[0]
                    pa_e = np.pi * np.prod(a) * np.sqrt((dirs_e ** 2 / a ** 2).sum(1))
                    r_pa_e = tensor_ratio(basis_e, pa_e)[0]
                    rows.append((r_pa, r_ell, r_pa_e, p.max() / p.min(), rms))
                r = np.array(rows).mean(0)
                print(f"{c:16s} {100 * (r[0] - 1):9.2f}% {100 * (r[1] - 1):9.2f}% "
                      f"{100 * (r[2] - 1):12.2f}% {r[3]:12.3f} {100 * r[4]:7.2f}%")


if __name__ == "__main__":
    main(*sys.argv[1:])
