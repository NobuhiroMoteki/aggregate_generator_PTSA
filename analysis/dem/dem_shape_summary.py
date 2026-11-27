#!/usr/bin/env python3
"""
Summarise the shape descriptors of the restructured aggregates (output of dem_shape.py).

All aggregates have N = 150 monomers of radius a.  With a single N, the fractal dimension and
prefactor are degenerate: one radius of gyration fixes only the curve k_f = N / (Rg_F/a)**Df.
The table therefore lists k_f at several fixed Df.  A Df from the slope of S(q) is printed as a
diagnostic, together with its value for the unrestructured aggregates (prescribed Df = 1.8).

Usage (from the repository root):
    python analysis/dem/dem_shape_summary.py out_fig.pdf \
        shape_flage.npz metrics_flage.npz [shape_tcca.npz metrics_tcca.npz]
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

N = 150
FRACS = [0, 15, 30, 50, 70, 80, 90, 95, 98, 100]
DF_GRID = [1.8, 2.0, 2.2, 2.4, 2.6, 2.8]
SQ_FIT = (2.0, 0.5)  # S(q) fit range: q Rg_F >= 2 and q a <= 0.5
NAMES = ["FLAGE (published inputs)", "TCCA (this repository)"]


def load(shape_path, metrics_path):
    s = dict(np.load(shape_path))
    m = np.load(metrics_path)
    key = {(int(a), int(f)): k for k, (a, f) in enumerate(zip(m["agg"], m["frac"]))}
    idx = np.array([key[(int(a), int(f))] for a, f in zip(s["agg"], s["frac"])])
    s["coord"] = m["coord"][idx]      # shape: (runs, dumps)
    s["conv3_c"] = m["conv3_c"][idx]  # shape: (runs, dumps)
    return s


def mci(v):
    v = np.asarray(v, dtype=float)
    return v.mean(), 1.96 * v.std(ddof=1) / np.sqrt(len(v))


def sq_slope_df(qa, sq, rgf):
    """Df = -slope of log S vs log q over SQ_FIT, per aggregate; shape of sq: (runs, nq)."""
    out = []
    for s, rg in zip(sq, rgf):
        m = (qa * rg >= SQ_FIT[0]) & (qa <= SQ_FIT[1])
        out.append(-np.polyfit(np.log(qa[m]), np.log(s[m]), 1)[0] if m.sum() >= 3 else np.nan)
    return np.array(out)


def table(name, d):
    print(f"\n=== {name}: {len(d['frac'])} runs ===")
    print("Rg_F includes the self term a^2; values in units of a; mean +- 95 % CI over aggregates")
    print("necks  Rg_F(49ns)      drift(39-49ns)  PA/a^2          PA/(N pi a^2)  "
          "sqrt(l1/l3)    l1/l2         coord          conv3")
    rows = [("init", np.ones(len(d["frac"]), bool), 0)]
    rows += [(f"{f}%", d["frac"] == f, 1) for f in FRACS]
    for lab, sel, st in rows:
        rg = d["rgf"][sel][:, 0 if st == 0 else -1]
        drift = (d["rgf"][sel][:, -1] - d["rgf"][sel][:, -11]) / d["rgf"][sel][:, -1]
        pa = d["ap"][sel][:, st]
        e = d["eig"][sel][:, st]
        cols = [mci(rg), mci(drift) if st else (np.nan, np.nan), mci(pa), mci(pa / (N * np.pi)),
                mci(np.sqrt(e[:, 0] / e[:, 2])), mci(e[:, 0] / e[:, 1]),
                mci(d["coord"][sel][:, 0 if st == 0 else -1]),
                mci(d["conv3_c"][sel][:, 0 if st == 0 else -1])]
        fmt = ["{:.3f}+-{:.3f}", "{:+.4f}+-{:.4f}", "{:.1f}+-{:.1f}", "{:.3f}+-{:.3f}",
               "{:.2f}+-{:.2f}", "{:.2f}+-{:.2f}", "{:.3f}+-{:.3f}", "{:.3f}+-{:.3f}"]
        print(f"{lab:6s} " + "  ".join(f.format(*c).ljust(14) for f, c in zip(fmt, cols)))

    print("\nk_f = N / (Rg_F/a)^Df at fixed Df (from the mean of k_f over aggregates)")
    print("necks  " + "  ".join(f"Df={x:.1f}".ljust(8) for x in DF_GRID) + "  Df from S(q) slope")
    qa = d["qa"]
    for lab, sel, st in rows:
        rg = d["rgf"][sel][:, 0 if st == 0 else -1]
        kf = [np.mean(N / rg ** x) for x in DF_GRID]
        dfs = sq_slope_df(qa, d["sq"][sel][:, st], rg)
        print(f"{lab:6s} " + "  ".join(f"{k:8.3f}" for k in kf)
              + f"  {np.nanmean(dfs):.2f}+-{1.96 * np.nanstd(dfs, ddof=1) / np.sqrt(len(dfs)):.2f}")


def main(fig_path, *paths):
    sets = [(NAMES[i], load(paths[2 * i], paths[2 * i + 1])) for i in range(len(paths) // 2)]
    for name, d in sets:
        table(name, d)

    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.6), constrained_layout=True)
    name, d = sets[0]
    cmap = plt.get_cmap("Blues")
    t = np.arange(d["rgf"].shape[1])  # dump index = time [ns]
    for k, f in enumerate((0, 50, 70, 90, 100)):
        sel = d["frac"] == f
        c = cmap(0.35 + 0.15 * k)
        ax[0].plot(t, d["rgf"][sel].mean(0), color=c, lw=1.6)
        ax[0].text(t[-1] + 0.6, d["rgf"][sel].mean(0)[-1], f"{f}%", fontsize=7.5, va="center")
        rg1 = d["rgf"][sel][:, -1].mean()
        ax[1].loglog(d["qa"] * rg1, d["sq"][sel][:, 1].mean(0), color=c, lw=1.6, label=f"{f}% necks")
    rg0 = d["rgf"][:, 0].mean()
    ax[1].loglog(d["qa"] * rg0, d["sq"][:, 0].mean(0), color="0.2", lw=1.2, ls="--",
                 label="initial ($D_\\mathrm{f}=1.8$)")
    ax[0].set_xlim(0, 55)
    ax[0].set_xlabel("Time [ns]")
    ax[0].set_ylabel(r"$R_{g,\mathrm{F}}/a$")
    ax[0].set_title("(a) radius of gyration, published inputs", fontsize=9.5, loc="left")
    ax[1].set_xlabel(r"$q R_{g,\mathrm{F}}$")
    ax[1].set_ylabel(r"$S(q)$")
    ax[1].set_title("(b) structure factor at 49 ns, published inputs", fontsize=9.5, loc="left")
    ax[1].legend(frameon=False, fontsize=7.5)
    style = {NAMES[0]: dict(color="#257de0", marker="o", ls="-"),
             NAMES[1]: dict(color="#f47230", marker="s", ls="--")}
    for name, d in sets:
        m, e = zip(*[mci(d["ap"][d["frac"] == f][:, 1] / (N * np.pi)) for f in FRACS])
        ax[2].errorbar(FRACS, m, yerr=e, capsize=2.5, lw=1.4, ms=4, label=name, **style[name])
        ax[2].axhline(np.mean(d["ap"][:, 0]) / (N * np.pi), color=style[name]["color"], lw=0.9,
                      ls=":")
    ax[2].set_xlabel("Necking fraction [%]")
    ax[2].set_ylabel(r"$\langle PA \rangle / (N \pi a^2)$")
    ax[2].set_title("(c) projected area at 49 ns (dotted: initial)", fontsize=9.5, loc="left")
    ax[2].legend(frameon=False, fontsize=7.5)
    for a in ax:
        a.grid(True, color="0.9", lw=0.6)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.savefig(fig_path)
    print(f"\nsaved {fig_path}")


if __name__ == "__main__":
    main(sys.argv[1], *sys.argv[2:])
