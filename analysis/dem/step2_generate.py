#!/usr/bin/env python3
"""
Build the aggregate set for comparing restructured aggregates with generator aggregates.

All aggregates have N = 150 monomers of radius a; coordinates are stored in units of a.
Classes:
  R/<set>/<frac>   restructured aggregates at 49 ns (set = flage or tcca, from dem_shape.py)
  I/<set>          the same aggregates before restructuring (Df = 1.8, k_f = 1.4)
  G/<method>/<Df>/<frac>
                   generator aggregates (method = tcca or ptsa) whose Rg_F equals the mean
                   Rg_F of R/flage/<frac>, i.e. k_f = N / (Rg_F/a)**Df, for several Df

Rg_F includes the self term a**2 (the convention of the tcca placement law).
A class whose generator fails for some seed keeps the successful aggregates only and the
failures are reported; a class without any success is not written.
Restructured aggregates with monomer overlaps beyond 1e-3 a are scaled uniformly to remove them
(remove_overlap); the number is stored as the attribute n_scaled.

Usage (from the repository root):
    python analysis/dem/step2_generate.py out.h5 [n_per_class]
"""
import json
import pathlib
import sys
import time

import h5py
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import tcca  # noqa: E402
from aggregate_ptsa_autogen_hdf5 import ptsa  # noqa: E402

N = 150
FRACS = [0, 50, 100]
DFS = [2.0, 2.2, 2.4, 2.6]
SEED_BASE = 2026112800
MAX_SEARCH = 100_000_000
HERE = pathlib.Path(__file__).resolve().parent


def rgf(xp):
    """Rg_F / a of monodisperse unit spheres at xp (shape (N, 3))."""
    return float(np.sqrt(((xp - xp.mean(0)) ** 2).sum(1).mean() + 1.0))


def remove_overlap(xs, dmin=1.999):
    """Scale each aggregate (shape (n, N, 3)) uniformly so that no centre distance is below 2.

    soot-dem leaves a few bonded pairs overlapping by up to 1.6 % of the diameter; MSTM needs
    non-overlapping spheres.  Overlaps of order 1e-4 a from the single-precision dumps are kept.
    Returns the scaled copy and the number of aggregates that were scaled.
    """
    out, n = xs.copy(), 0
    for k, x in enumerate(xs):
        d = np.sqrt(((x[:, None] - x[None]) ** 2).sum(-1))[np.triu_indices(len(x), 1)]
        if d.min() < dmin:
            c = x.mean(0)
            out[k] = c + (x - c) * (2.0 / d.min())
            n += 1
    return out, n


def generate(method, Df, kf, seed):
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    if method == "tcca":
        r = tcca.tcca(N, 1.0, 0.0, Df, kf, rng=rng)
    else:
        r = ptsa(N, 1.0, 0.0, Df, kf, MAX_SEARCH, rng)
    if r[0] != N:
        return None
    xp = np.asarray(r[1], dtype=float)  # shape: (N, 3)
    # Contact window of each generator: tcca 1e-2 a; ptsa grows with the monomer index up to
    # tol * (1 + 5 (2.95 - Df) / 0.6) a with tol = 0.001 + 0.0033 (2.95 - Df).
    if method == "tcca":
        max_sep = 1e-2
    else:
        max_sep = (0.001 + 0.0033 * (2.95 - Df)) * (1 + 5 * (2.95 - Df) / 0.6) * 1.001
    if not tcca.check_geometry(xp, np.ones(N), max_sep)["ok"]:
        return None
    return xp - xp.mean(0)


def main(out, n_per_class=50):
    n_per_class = int(n_per_class)
    classes = {}  # name -> (xp array (n, N, 3), attrs)
    for s in ("flage", "tcca"):
        d = np.load(HERE / f"shape_{s}.npz")
        first = d["frac"] == 0
        xs, n = remove_overlap(d["xp0"][first])
        classes[f"I/{s}"] = (xs, dict(kind="initial", set=s, n_scaled=n))
        for f in FRACS:
            sel = d["frac"] == f
            xs, n = remove_overlap(d["xp1"][sel])
            classes[f"R/{s}/{f}"] = (xs, dict(kind="restructured", set=s, frac=f, n_scaled=n))
            print(f"R/{s}/{f}: {n} aggregates scaled to remove overlaps")
    d = np.load(HERE / "shape_flage.npz")
    target = {f: float(d["rgf"][d["frac"] == f][:, -1].mean()) for f in FRACS}
    print("target Rg_F/a:", {f: round(v, 3) for f, v in target.items()})

    tcca.tcca(20, 1.0, 0.0, 2.0, 1.0, rng=np.random.default_rng(0))  # JIT warm-up
    ptsa(20, 1.0, 0.0, 2.0, 1.0, 1000, np.random.default_rng(0))
    report = []
    for im, method in enumerate(("tcca", "ptsa")):
        for idf, Df in enumerate(DFS):
            for jf, f in enumerate(FRACS):
                kf = N / target[f] ** Df
                t0 = time.time()
                xs, fails = [], 0
                for i in range(n_per_class):
                    seed = SEED_BASE + 100000 * im + 10000 * idf + 1000 * jf + i
                    xp = generate(method, Df, kf, seed)
                    if xp is None:
                        fails += 1
                    else:
                        xs.append(xp)
                xs = np.array(xs)
                rg = np.array([rgf(x) for x in xs]) if len(xs) else np.array([np.nan])
                name = f"G/{method}/{Df:.1f}/{f}"
                classes[name] = (xs, dict(kind="generator", method=method, Df=Df, kf=kf, frac=f,
                                          rgf_target=target[f], n_fail=fails))
                line = (f"{name:16s} kf={kf:.4f}  n={len(xs)}  fails={fails}  "
                        f"Rg_F={rg.mean():.3f}+-{rg.std():.3f} (target {target[f]:.3f})  "
                        f"{time.time() - t0:.1f} s")
                print(line, flush=True)
                report.append(line)

    with h5py.File(out, "w") as h:
        h.attrs["N"] = N
        h.attrs["units"] = "monomer radius a"
        classes = {k: v for k, v in classes.items() if len(v[0])}  # infeasible classes are dropped
        h.attrs["classes"] = json.dumps(list(classes))
        h.attrs["report"] = "\n".join(report)
        for name, (xs, attrs) in classes.items():
            g = h.create_group(name)
            g.create_dataset("xp", data=xs)  # shape: (n, N, 3)
            for k, v in attrs.items():
                g.attrs[k] = v
    print(f"saved {out}: {len(classes)} classes")


if __name__ == "__main__":
    main(*sys.argv[1:])
