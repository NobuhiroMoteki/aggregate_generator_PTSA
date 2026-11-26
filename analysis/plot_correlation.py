#!/usr/bin/env python3
"""
Fit correlation dimensions and plot C(r) from correlation_compare.py output.

For each aggregate, log C is regressed on log r over bins whose centres lie in
[r_lo * a, f_hi * R_F].  The correlation dimension is D_c = 3 + slope
(Filippov et al. 2000, Eq. [6]).  Several fit ranges are reported to show the
sensitivity to this choice.

Usage (run from the repository root):
    python analysis/plot_correlation.py analysis/correlation_results.npz docs/figs/correlation.pdf
"""

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Fixed categorical order (colour follows the method in every panel);
# line style is a second encoding for print and colour-vision deficiency.
STYLE = {
    "ptsa": dict(color="#257de0", ls="-", label="PTSA"),
    "tcca": dict(color="#f47230", ls="--", label="TCCA"),
    "sa": dict(color="#00b894", ls=":", label="SA (Filippov)"),
}
FIT_RANGES = [(5.0, 0.5), (4.0, 0.4), (5.0, 0.7), (8.0, 0.7)]  # (r_lo / a, f_hi)
PANELS = [("filippov", 1.8, r"$D_\mathrm{f}=1.8,\ k_\mathrm{f}=1.3$, monodisperse"),
          ("project", 2.35, r"$D_\mathrm{f}=2.35,\ k_\mathrm{f}=0.95,\ \sigma_a=0.1$"),
          ("project", 2.55, r"$D_\mathrm{f}=2.55,\ k_\mathrm{f}=0.95,\ \sigma_a=0.1$")]


def slope(r, C, lo, hi):
    """Least-squares slope of log C vs log r for lo <= r <= hi and C > 0."""
    m = (r >= lo) & (r <= hi) & (C > 0)
    if m.sum() < 3:
        return np.nan
    return np.polyfit(np.log(r[m]), np.log(C[m]), 1)[0]


def main(path, fig_path):
    d = np.load(path)
    edges = d["edges"]
    r = np.sqrt(edges[1:] * edges[:-1])  # geometric bin centres, shape: (n_bins,)
    case, meth, Df, RF, C = d["case"], d["meth"], d["Df"], d["RF"], d["C"]
    print(f"failed aggregates: {int(d['n_failed'])}")

    print("\nCorrelation dimension D_c = 3 + slope, mean +- standard error")
    head = "case      Df    method  n   " + "  ".join(
        f"[{lo:g}a,{hi:g}RF]" for lo, hi in FIT_RANGES) + "   <RF>/a  <time>/s"
    print(head)
    for cs, df, _ in PANELS:
        for mt in ("sa", "ptsa", "tcca"):
            sel = (case == cs) & (Df == df) & (meth == mt)
            if not sel.any():
                continue
            cols = []
            for lo, hi in FIT_RANGES:
                dc = np.array([3 + slope(r, c, lo, hi * rf)
                               for c, rf in zip(C[sel], RF[sel])])
                cols.append(f"{dc.mean():.3f}+-{dc.std(ddof=1) / np.sqrt(len(dc)):.3f}")
            print(f"{cs:9s} {df:4.2f}  {mt:6s} {sel.sum():3d}   " + "  ".join(cols)
                  + f"   {RF[sel].mean():6.1f}  {d['sec'][sel].mean():6.1f}")

    # Local correlation dimension D_c(r) = 3 + d ln C / d ln r of the mean C(r),
    # from a least-squares slope over a sliding window of WIN bins.
    WIN = 7
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), constrained_layout=True,
                             sharey=True)
    for ax, (cs, df, title) in zip(axes, PANELS):
        for mt in ("sa", "ptsa", "tcca"):
            sel = (case == cs) & (Df == df) & (meth == mt)
            if not sel.any():
                continue
            Cm = C[sel].mean(axis=0)
            rf = RF[sel].mean()
            x, y = [], []
            for i0 in range(len(r) - WIN + 1):
                rr, cc = r[i0:i0 + WIN], Cm[i0:i0 + WIN]
                if np.all(cc > 0):
                    x.append(np.exp(np.log(rr).mean()) / rf)
                    y.append(3 + np.polyfit(np.log(rr), np.log(cc), 1)[0])
            ax.plot(x, y, lw=1.8, zorder=3 if mt == "sa" else 2, **STYLE[mt])
        ax.axhline(df, color="0.45", lw=1.0, ls="-.",
                   label=r"prescribed $D_\mathrm{f}$")
        ax.set_xscale("log")
        ax.set_xlim(0.05, 1.5)
        ax.set_ylim(0.8, 3.0)
        ax.set_xlabel(r"$r/R_\mathrm{F}$")
        ax.set_title(title, fontsize=9.5)
        ax.grid(True, which="major", color="0.88", lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel(r"$D_\mathrm{c}(r) = 3 + \mathrm{d}\ln C/\mathrm{d}\ln r$")
    axes[0].legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.savefig(fig_path)
    print(f"\nsaved {fig_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
