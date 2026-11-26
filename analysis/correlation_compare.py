#!/usr/bin/env python3
"""
Two-point correlation function C(r) of aggregates from SA, PTSA and TCCA.

Case "filippov": Df = 1.8, kf = 1.3, monodisperse, Np = 2048 (the parameters of
Filippov et al. 2000, Fig. 1).  "sa" is the pure sequential algorithm of
Filippov et al. (tcca with n_SA = Np).
Case "project": k = 0.95, rel_std_rp = 0.10, Np = 2048, Df = 2.35 and 2.55
(the same seeds as the structure comparison in the README).

The local slope of log C versus log r in the range [r_lo, f_hi * R_F] gives
the correlation dimension D_c = 3 + slope (Filippov et al. 2000, Eq. [6]).

Usage (run from the repository root):
    python analysis/correlation_compare.py --workers 20 --out analysis/correlation_results.npz
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import tcca  # noqa: E402
from aggregate_ptsa_autogen_hdf5 import ptsa  # noqa: E402

NP = 2048
# bin edges in units of the mean monomer radius; shape: (N_EDGE,)
EDGES = np.geomspace(1.9, 400.0, 81)


def cases():
    out = []
    for s in range(40):
        for meth in ("sa", "ptsa", "tcca"):
            out.append(("filippov", meth, 1.8, 1.3, 0.0, 3000 + s))
    for Df in (2.35, 2.55):
        for s in range(30):
            for meth in ("ptsa", "tcca"):
                out.append(("project", meth, Df, 0.95, 0.10, 500 + s))
    return out


def run(c):
    case, meth, Df, k, rs, seed = c
    a = 1.0  # mean monomer radius (lengths in units of a)
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    if meth == "ptsa":
        r = ptsa(NP, a, rs, Df, k, 10**8, rng)
    elif meth == "tcca":
        r = tcca.tcca(NP, a, rs, Df, k, rng=rng)
    else:
        r = tcca.tcca(NP, a, rs, Df, k, n_SA=NP, max_search=10**8, rng=rng)
    sec = time.perf_counter() - t0
    if r[0] != NP:
        return dict(case=case, meth=meth, Df=Df, seed=seed, ok=False, sec=sec)
    xp, rp = r[1], r[2]
    mp = rp ** 3
    xc = xp - np.sum(mp[:, None] * xp, axis=0) / np.sum(mp)
    RF = np.sqrt(np.sum(mp * (np.sum(xc ** 2, axis=1) + rp ** 2)) / np.sum(mp))
    return dict(case=case, meth=meth, Df=Df, seed=seed, ok=True, sec=sec,
                C=tcca.pair_correlation(xp, EDGES), RF=RF)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--out", default="analysis/correlation_results.npz")
    args = p.parse_args()
    res = []
    with ProcessPoolExecutor(args.workers) as ex:
        for r in ex.map(run, cases()):
            print(r["case"], r["meth"], r["Df"], r["seed"], r["ok"],
                  f"{r['sec']:.1f}s", flush=True)
            res.append(r)
    ok = [r for r in res if r["ok"]]
    np.savez(args.out,
             edges=EDGES,
             case=np.array([r["case"] for r in ok]),
             meth=np.array([r["meth"] for r in ok]),
             Df=np.array([r["Df"] for r in ok]),
             seed=np.array([r["seed"] for r in ok]),
             RF=np.array([r["RF"] for r in ok]),
             C=np.array([r["C"] for r in ok]),       # shape: (n_agg, n_bins)
             sec=np.array([r["sec"] for r in ok]),
             n_failed=len(res) - len(ok))
    print(f"saved {args.out}: {len(ok)} aggregates, {len(res) - len(ok)} failed")


if __name__ == "__main__":
    main()
