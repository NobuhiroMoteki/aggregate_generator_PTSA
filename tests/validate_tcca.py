#!/usr/bin/env python3
"""
Validation of tcca.py (plain script, no pytest needed).

Run from the repository root:
    python tests/validate_tcca.py

Checks
------
1. Schedule for Np = n_SA * 2**L equals the C++ hierarchy.
2. Monodisperse limit: placement distances equal the C++ formulas.
3. Geometry: no overlap, every sphere has a neighbour in contact,
   contact graph connected.
4. Fractal law: every merged sub-cluster obeys N = k*(Rg_F/a)**Df (rtol 1e-10).
5. Reproducibility: same seed -> identical output; cell list == brute force.
6. Failure is reported (Np_final < Np) and does not hang.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import tcca  # noqa: E402

FAILED = []


def check(name, cond, info=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {info}")
    if not cond:
        FAILED.append(name)


def _split_ops(ops):
    leaves = ops[ops[:, 0] == 0][:, [1, 3]].copy()
    leaves[:, 1] -= leaves[:, 0]               # rows (start, n)
    merges = ops[ops[:, 0] == 1][:, 1:]        # rows (start, mid, end)
    return leaves, merges


def test_schedule():
    ops, sub = tcca.make_schedule(4 * 2 ** 5, 4)
    leaves, merges = _split_ops(ops)
    check("schedule: leaves all n_SA", np.all(leaves[:, 1] == 4) and len(leaves) == 32)
    check("schedule: equal halves", np.all(merges[:, 1] - merges[:, 0]
                                           == merges[:, 2] - merges[:, 1]))
    check("schedule: n_merge = n_leaf - 1", len(merges) == len(leaves) - 1)
    for Np in (1, 2, 5, 350, 2800, 21720):
        ops, sub = tcca.make_schedule(Np, 4)
        leaves, merges = _split_ops(ops)
        cover = np.zeros(Np, dtype=int)
        for s, n in leaves:
            cover[s:s + n] += 1
        ok = np.all(cover == 1) and np.all(leaves[:, 1] <= 4)
        ok &= (len(merges) == 0 and Np <= 4) or tuple(merges[-1]) [::2] == (0, Np)
        # each subtree is contiguous and starts at its leftmost monomer
        ok &= all(ops[sub[i], 1] == ops[i, 1] and np.all(ops[sub[i]:i + 1, 3] <= ops[i, 3])
                  for i in range(len(ops)))
        check(f"schedule: partition Np={Np}", bool(ok))


def test_monodisperse_formula():
    a, kf, Df = 1.0, 1.3, 1.8
    rng = np.random.default_rng(0)
    for _ in range(5):
        N1 = int(rng.integers(4, 2000))
        N2 = N1
        R1_2, R2_2 = rng.uniform(1, 50, 2)
        m = 4 * np.pi / 3
        g = tcca._gamma2(m * N1, m * N2, R1_2, R2_2, N1 + N2, a * a, kf, Df)
        # C++ aggregate_gen_CCA (N1 == N2, so integer divisions are exact)
        cpp = (a * a * (N1 + N2) * (N1 + N2) / (N1 * N2)
               * ((N1 + N2) / kf) ** (2 / Df)
               - (N1 + N2) // N2 * R1_2 - (N1 + N2) // N1 * R2_2)
        check(f"CCA formula == C++ (N1=N2={N1})", np.isclose(g, cpp, rtol=1e-12))
    for N in (3, 4, 10, 100):
        m = 4 * np.pi / 3
        g = tcca._gamma2(m * (N - 1), m, a * a * ((N - 1) / kf) ** (2 / Df),
                         a * a, N, a * a, kf, Df)
        cpp = (N * N * a * a / (N - 1) * (N / kf) ** (2 / Df)
               - N * a * a / (N - 1) - N * a * a * ((N - 1) / kf) ** (2 / Df))
        check(f"SA formula == C++ (N={N})", np.isclose(g, cpp, rtol=1e-12))


def _subcluster_keff(xp, rp, merges, Df, a):
    keff = []
    for s, _, e in merges:
        keff.append(tcca.fractal_prefactor_filippov(xp[s:e], rp[s:e], Df))
    return np.array(keff)


def test_generation():
    mean_rp = 0.015
    tol = 1e-2
    cases = [  # (Np, rel_std_rp, Df, k)
        (128, 0.0, 1.8, 1.3),
        (300, 0.0, 2.5, 1.0),
        (350, 0.10, 2.0, 1.2),
        (500, 0.10, 2.5, 0.95),
        (1000, 0.10, 2.3, 0.95),
        (2000, 0.10, 2.0, 0.95),   # low Df: needs sub-cluster rebuilds
        (1024, 0.10, 1.8, 1.3),
    ]
    for Np, rs, Df, k in cases:
        t0 = time.perf_counter()
        out = tcca.tcca(Np, mean_rp, rs, Df, k, tol=tol,
                        rng=np.random.default_rng(42))
        dt = time.perf_counter() - t0
        Np_final, xp, rp = out[:3]
        label = f"Np={Np}, rel_std={rs}, Df={Df}, k={k}"
        check(f"generated  {label}", Np_final == Np, f"({dt:.2f} s)")
        if Np_final != Np:
            continue
        g = tcca.check_geometry(xp, rp, tol * mean_rp)
        check(f"geometry   {label}", g["ok"] and g["max_nn_gap"] < tol * mean_rp,
              f"min_gap/a={g['min_gap'] / mean_rp:.2e}, "
              f"max_nn_gap/a={g['max_nn_gap'] / mean_rp:.2e}, "
              f"components={g['n_components']}")
        _, merges = _split_ops(tcca.make_schedule(Np, 4)[0])
        keff = _subcluster_keff(xp, rp, merges, Df, None)
        err = np.max(np.abs(keff / k - 1))
        check(f"fractal law {label}", err < 1e-10,
              f"(max |k_eff/k - 1| over {len(merges)} merges = {err:.1e})")


def test_reproducibility():
    args = (700, 0.015, 0.10, 2.2, 1.1)
    a = tcca.tcca(*args, rng=np.random.default_rng(7))
    b = tcca.tcca(*args, rng=np.random.default_rng(7))
    c = tcca.tcca(*args, rng=np.random.default_rng(7), use_cells=False)
    check("same seed -> identical", a[0] == 700 and np.array_equal(a[1], b[1]))
    check("cell list == brute force (bit-identical)",
          np.array_equal(a[1], c[1]) and np.array_equal(a[2], c[2]))
    seed = np.random.SeedSequence()
    d = tcca.tcca(*args, rng=np.random.default_rng(seed))
    e = tcca.tcca(*args, rng=np.random.default_rng(
        np.random.SeedSequence(int(str(seed.entropy)))))
    check("reproducible from stored seed_entropy string", np.array_equal(d[1], e[1]))


def test_pair_correlation():
    out = tcca.tcca(300, 1.0, 0.1, 2.0, 1.2, rng=np.random.default_rng(3))
    edges = np.linspace(0.0, 200.0, 2001)
    edges[0] = 1e-9
    C = tcca.pair_correlation(out[1], edges)
    integral = np.sum(C * 4 / 3 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3))
    check("pair correlation normalisation = (N-1)/2",
          np.isclose(integral, (300 - 1) / 2, rtol=1e-12), f"({integral})")


def test_failure():
    t0 = time.perf_counter()
    out = tcca.tcca(256, 0.015, 0.0, 2.95, 0.95, iter1max=20, iter2max=200,
                    rng=np.random.default_rng(1))
    dt = time.perf_counter() - t0
    check("failure reported (Df=2.95, k=0.95)", out[0] < 256 and out[4] == 0.0,
          f"(Np_final={out[0]}, {dt:.2f} s)")


if __name__ == "__main__":
    t0 = time.perf_counter()
    tcca.tcca(20, 0.015, 0.1, 2.0, 1.0, rng=np.random.default_rng(0))
    print(f"JIT compile: {time.perf_counter() - t0:.1f} s")
    test_schedule()
    test_monodisperse_formula()
    test_generation()
    test_reproducibility()
    test_pair_correlation()
    test_failure()
    print()
    if FAILED:
        print(f"{len(FAILED)} check(s) FAILED: {FAILED}")
        sys.exit(1)
    print("All checks passed.")
