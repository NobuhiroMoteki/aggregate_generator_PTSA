#!/usr/bin/env python3
"""
Compare restructured and generator aggregates at equal N and equal mobility diameter.

For each compact generator series M/<method>/<Df>/* (step2b_generate.py), the class means of
d_mob, PA, R_S and the optical quantities are interpolated linearly in Rg_F to the Rg_F* at
which d_mob equals that of the restructured class R/flage/f.  The table lists the % difference
of the interpolated generator values from the restructured ones.

Usage (from the repository root):
    python analysis/dem/step2b_summary.py step2_optics.csv step2_mobility.csv \
        step2b_optics.csv step2b_mobility.csv
"""
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from step2_summary import collect, read_csv  # noqa: E402

FRACS = [0, 50, 100]
SERIES = [("ptsa", 2.6), ("ptsa", 2.2), ("tcca", 2.2)]
COND = [(0.453, 1.95, 0.79), (0.638, 1.95, 0.79), (0.834, 1.95, 0.79), (0.638, 1.75, 0.60)]


def main(opt_a, mob_a, opt_b, mob_b):
    ref = collect(read_csv(opt_a), read_csv(mob_a))
    gen = collect(read_csv(opt_b), read_csv(mob_b))
    keys = ["PA_um2", "R_S_um", "coord_005"]
    labels = ["PA", "R_S", "coord"]
    for wl, mr, mi in COND:
        tag = f"{wl:.3f}_{mr:.2f}_{mi:.2f}"
        keys += [f"Qabs_{tag}", f"Qsca_{tag}", f"ImS_{tag}"]
        labels += [f"Qabs {wl}/{mr}", f"Qsca {wl}/{mr}", f"|ImS| {wl}/{mr}"]
    print("Generator interpolated to the d_mob of the restructured class; "
          "% difference from restructured (coord: generator value)")
    for f in FRACS:
        r = ref[f"R/flage/{f}"]
        d0 = r["d_mob_um"].mean()
        print(f"\n--- R/flage/{f}: d_mob = {d0:.4f} um, Rg_F/a = {r['rgf'].mean():.3f}, "
              f"coord = {r['coord_005'].mean():.2f} ---")
        print(f"{'series':12s} {'Rg_F*':>6s} " + " ".join(f"{lab:>14s}" for lab in labels))
        for meth, Df in SERIES:
            names = sorted((n for n in gen if n.startswith(f"M/{meth}/{Df:.1f}/")),
                           key=lambda n: float(n.split("/")[-1]))
            rg = np.array([gen[n]["rgf"].mean() for n in names])
            dm = np.array([gen[n]["d_mob_um"].mean() for n in names])
            if not (dm.min() <= d0 <= dm.max()):
                print(f"{meth}/{Df:.1f}    d_mob {d0:.4f} outside the generated range "
                      f"[{dm.min():.4f}, {dm.max():.4f}]")
                continue
            rg_star = np.interp(d0, dm, rg)
            out = []
            for k in keys:
                g = np.interp(rg_star, rg, [gen[n][k].mean() for n in names])
                out.append(f"{g:.2f}" if k == "coord_005" else f"{100 * (g / r[k].mean() - 1):+.1f}%")
            print(f"{meth}/{Df:.1f}     {rg_star:6.2f} " + " ".join(f"{o:>14s}" for o in out))


if __name__ == "__main__":
    main(*sys.argv[1:])
