#!/usr/bin/env python3
"""
Compact generator aggregates for matching the restructured aggregates in mobility diameter.

The Rg_F-matched generator classes of step2_generate.py have a 3-17 % larger mobility diameter
than the restructured aggregates.  This script builds generator classes with smaller Rg_F so
that the restructured d_mob can be bracketed and the optics compared at equal N and d_mob.
Classes: M/<method>/<Df>/<Rg_F>, N = 150, coordinates in units of a.

Usage (from the repository root):
    python analysis/dem/step2b_generate.py out.h5 [n_per_class]
"""
import json
import pathlib
import sys
import time

import h5py
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from step2_generate import N, generate, rgf  # noqa: E402

RGF = [6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 10.0, 10.5, 11.0]  # target Rg_F / a
GENERATORS = [("ptsa", 2.6), ("ptsa", 2.2), ("tcca", 2.2)]
SEED_BASE = 2026112900


def main(out, n_per_class=30):
    n_per_class = int(n_per_class)
    classes, report = {}, []
    generate("tcca", 2.0, 1.0, 0)  # JIT warm-up
    generate("ptsa", 2.0, 1.0, 0)
    for ig, (method, Df) in enumerate(GENERATORS):
        for ir, rg in enumerate(RGF):
            kf = N / rg ** Df
            t0 = time.time()
            xs, fails = [], 0
            for i in range(n_per_class):
                xp = generate(method, Df, kf, SEED_BASE + 100000 * ig + 1000 * ir + i)
                if xp is None:
                    fails += 1
                    if fails >= 5 and not xs:  # infeasible: stop early
                        break
                else:
                    xs.append(xp)
            name = f"M/{method}/{Df:.1f}/{rg:.1f}"
            line = f"{name:18s} kf={kf:.4f} n={len(xs)} fails={fails} {time.time() - t0:.1f} s"
            if xs:
                classes[name] = (np.array(xs), dict(kind="compact", method=method, Df=Df, kf=kf,
                                                    rgf_target=rg, n_fail=fails))
                line += f"  Rg_F={np.mean([rgf(x) for x in xs]):.3f}"
            print(line, flush=True)
            report.append(line)
    with h5py.File(out, "w") as h:
        h.attrs["N"] = N
        h.attrs["units"] = "monomer radius a"
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
