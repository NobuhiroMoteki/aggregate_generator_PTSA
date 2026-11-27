#!/usr/bin/env python3
"""
Summarise soot-dem restructuring metrics and compare with Demidov et al. (2024), Fig. 8.

Usage (from the repository root):
    python analysis/dem/dem_summary.py out_fig.pdf analysis/dem/metrics_flage.npz \
        [analysis/dem/metrics_tcca.npz]
Each npz is written by dem_metrics.py.  Terminal values are taken at the last dump (t = 49 ns).
Error bars are 95 % confidence intervals of the mean (1.96 standard errors).
With two npz files, the differences (second minus first) of the terminal values and of the
increments during the run (terminal minus initial) are also printed.
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FRACS = [0, 15, 30, 50, 70, 80, 90, 95, 98, 100]
# Values read off Fig. 8(b) and 8(d) of Demidov et al. (2024); uncertainty about +-0.05 and +-0.01.
PAPER_COORD = [4.85, 4.65, 4.40, 3.90, 3.35, 3.00, 2.65, 2.45, 2.35, 2.30]
PAPER_CONV = [0.300, 0.295, 0.290, 0.270, 0.235, 0.210, 0.180, 0.160, 0.150, 0.145]
STYLE = {"FLAGE (published inputs)": dict(color="#257de0", ls="-", marker="o"),
         "TCCA (this repository)": dict(color="#f47230", ls="--", marker="s")}
DT_NS = 1.0  # time between dumps [ns]


def load(path):
    d = np.load(path)
    return {k: d[k] for k in d.files}


def terminal(d, key, frac):
    sel = d["frac"] == frac
    v = d[key][sel]
    v = v[:, -1] if v.ndim == 2 and key != "conv2_1" else v
    return v


def mean_ci(v):
    return v.mean(), 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan


def table(name, d):
    print(f"\n{name}: {len(d['frac'])} runs")
    print("frac  n   coord(ours)      coord(paper)  conv3(ours)       conv3(paper)  "
          "conv2 hull-discs  conv2 hull-centres")
    for f, pc, pv in zip(FRACS, PAPER_COORD, PAPER_CONV):
        sel = d["frac"] == f
        if not sel.any():
            continue
        c = mean_ci(d["coord"][sel][:, -1])
        v = mean_ci(d["conv3_c"][sel][:, -1])
        w1 = mean_ci(d["conv2_1"][sel][:, 0])
        w2 = mean_ci(d["conv2_1"][sel][:, 1])
        print(f"{f:4d} {sel.sum():3d}   {c[0]:.3f}+-{c[1]:.3f}    {pc:.2f}         "
              f"{v[0]:.3f}+-{v[1]:.3f}    {pv:.3f}         {w1[0]:.3f}+-{w1[1]:.3f}    "
              f"{w2[0]:.3f}+-{w2[1]:.3f}")
    c0 = mean_ci(d["coord"][:, 0])
    v0 = mean_ci(d["conv3_c"][:, 0])
    print(f"initial: coord {c0[0]:.3f}+-{c0[1]:.3f} (paper ~2.05-2.1), "
          f"conv3 {v0[0]:.3f}+-{v0[1]:.3f} (paper ~0.105)")


def compare(d1, d2):
    """Print d2 - d1 per necking fraction with the 95 % CI and z = difference / standard error."""
    for key, lab in (("coord", "coordination"), ("conv3_c", "3D convexity")):
        print(f"\n{lab}: TCCA - FLAGE, mean +- 1.96 SE (z)")
        print("frac   terminal                  increment (terminal - initial)")
        for f in FRACS:
            out = []
            for inc in (False, True):
                se2, diff = 0.0, 0.0
                for sgn, d in ((-1, d1), (1, d2)):
                    v = d[key][d["frac"] == f]
                    v = v[:, -1] - v[:, 0] if inc else v[:, -1]
                    diff += sgn * v.mean()
                    se2 += v.var(ddof=1) / len(v)
                se = np.sqrt(se2)
                out.append(f"{diff:+.4f} +- {1.96 * se:.4f} ({diff / se:+.1f})")
            print(f"{f:4d}   " + "     ".join(out))


def main(fig_path, *paths):
    sets = list(zip(STYLE.keys(), [load(p) for p in paths]))
    for name, d in sets:
        table(name, d)
    if len(sets) == 2:
        compare(sets[0][1], sets[1][1])

    fig, ax = plt.subplots(2, 2, figsize=(9.5, 6.5), constrained_layout=True)
    # (a), (c): time evolution at selected necking fractions, published inputs only
    name, d = sets[0]
    for f in (0, 50, 70, 90, 100):
        sel = d["frac"] == f
        if not sel.any():
            continue
        t = np.arange(d["coord"].shape[1]) * DT_NS
        sel_a = sel & (d["agg"] < 20)  # Fig. 8(a) averages 20 aggregates
        ax[0, 0].plot(t, d["coord"][sel_a].mean(0), color="0.25", lw=1.4)
        ax[0, 0].text(t[-1] + 0.5, d["coord"][sel_a].mean(0)[-1], f"{f}%", fontsize=7.5,
                      va="center")
        ax[1, 0].plot(t, d["conv3_c"][sel].mean(0), color="0.25", lw=1.4)
        ax[1, 0].text(t[-1] + 0.5, d["conv3_c"][sel].mean(0)[-1], f"{f}%", fontsize=7.5,
                      va="center")
    ax[0, 0].set_ylabel("Coordination number")
    ax[1, 0].set_ylabel("Convexity (3D)")
    ax[1, 0].set_xlabel("Time [ns]")
    ax[0, 0].set_title("(a) published inputs, mean of 20", fontsize=9.5, loc="left")
    ax[1, 0].set_title("(c) published inputs, mean of 50", fontsize=9.5, loc="left")
    ax[0, 0].set_xlim(0, 56)
    ax[1, 0].set_xlim(0, 56)

    # (b), (d): terminal values versus necking fraction
    for key, axx, paper, lab in (("coord", ax[0, 1], PAPER_COORD, "(b)"),
                                 ("conv3_c", ax[1, 1], PAPER_CONV, "(d)")):
        axx.plot(FRACS, paper, color="0.45", ls=":", marker="x", ms=6, lw=1.2,
                 label="Demidov et al. (2024), read off Fig. 8")
        for name, d in sets:
            m, e = zip(*[mean_ci(d[key][d["frac"] == f][:, -1]) if (d["frac"] == f).any()
                         else (np.nan, np.nan) for f in FRACS])
            axx.errorbar(FRACS, m, yerr=e, capsize=2.5, lw=1.4, ms=4, label=name,
                         **STYLE[name])
        axx.set_title(f"{lab} terminal value (t = 49 ns)", fontsize=9.5, loc="left")
    ax[0, 1].legend(frameon=False, fontsize=8)
    ax[1, 1].set_xlabel("Necking fraction [%]")
    for a in ax.flat:
        a.grid(True, color="0.9", lw=0.6)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.savefig(fig_path)
    print(f"\nsaved {fig_path}")


if __name__ == "__main__":
    main(sys.argv[1], *sys.argv[2:])
