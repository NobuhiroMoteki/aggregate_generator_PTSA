#!/usr/bin/env python3
"""
Bound on the harmonic-mean (mean-mobility) effect in the free molecular term of the mobility
diameter, from the equivalent ellipsoid.

The adjusted-sphere expression with the orientation-averaged projected area PA gives the
arithmetic mean of the free molecular friction over orientations.  A DMA measures the mean of
the mobility, i.e. the harmonic mean of the friction.  The continuum limit R_S (Hubbard-Douglas)
already corresponds to the harmonic mean.  Here PA is replaced by PA_H = PA / (f_A / f_H), where
f_A / f_H is the ratio of the arithmetic to the harmonic mean of the free molecular friction
tensor of the ellipsoid with the same gyration tensor (semi-axes sqrt(5) g_k).

The ellipsoid treats the aggregate as a solid convex body.  For open aggregates it overestimates
f_A / f_H: a projected-area tensor that includes the shielding between monomers gives f_A/f_H - 1
3-6 times smaller for the restructured and the Df = 1.8 aggregates (step2_pa_tensor.py).  d_mob_H is therefore a bound on the effect, not a
correction; the exact f_H needs the free molecular friction tensor of the aggregate itself.

The external modules are given by path:
  ANISOTROPY_MODULE must provide principal_radii(pos, rad) -> (g1, g2, g3) and
                    friction_ratio_ellipsoid(semi) -> (f_A / f_H, K)
  MOBILITY_MODULE   must provide mobility_diameter(r_s, pa) -> (d_mob, Kn)

Usage:
    python step2_harmonic.py aggregates.h5 mobility.csv ANISOTROPY_MODULE.py MOBILITY_MODULE.py out.csv
Writes mobility.csv with the columns fa_fh, PA_H_um2, d_mob_H_um appended.
"""
import csv
import importlib.util
import json
import sys

import h5py
import numpy as np

A_UM = 0.015  # monomer radius [um]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(h5path, mob_csv, aniso_path, mob_path, out):
    an = load(aniso_path, "anisotropy_model")
    mm = load(mob_path, "mobility_model")
    xp = {}
    with h5py.File(h5path, "r") as h:
        for name in json.loads(h.attrs["classes"]):
            xp[name] = h[name]["xp"][:]  # shape: (n, N, 3), units of a
    rows = list(csv.DictReader(open(mob_csv)))
    for r in rows:
        x = xp[r["class"]][int(r["idx"])] * A_UM
        g = an.principal_radii(x, np.full(len(x), A_UM))
        fa_fh = an.friction_ratio_ellipsoid(np.sqrt(5.0) * g)[0]
        pa_h = float(r["PA_um2"]) / fa_fh
        d_h, _ = mm.mobility_diameter(float(r["R_S_um"]), pa_h)
        r.update(fa_fh=f"{fa_fh:.6f}", PA_H_um2=f"{pa_h:.8e}", d_mob_H_um=f"{d_h:.8e}")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {out}: {len(rows)} aggregates")


if __name__ == "__main__":
    main(*sys.argv[1:])
