#!/usr/bin/env python3
"""
Compare restructured aggregates with generator aggregates of the same N and Rg_F.

Reads the outputs of step2_mstm.jl (optics) and step2_mobility.py (mobility, structure).
For each target necking fraction f, every class is compared with the restructured class
R/flage/f: relative difference of the class means in % and z = difference / standard error.

Usage (from the repository root):
    python analysis/dem/step2_summary.py optics.csv mobility.csv out_fig.pdf
"""
import csv
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FRACS = [0, 50, 100]
DFS = [2.0, 2.2, 2.4, 2.6]
WL0, M0 = 0.638, (1.95, 0.79)  # reference optical condition for the table


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def collect(optics, mobility):
    """Return {class: {quantity: array over aggregates}}."""
    data = defaultdict(lambda: defaultdict(list))
    for r in mobility:
        for k in ("rgf", "R_S_um", "PA_um2", "d_mob_um", "coord_005", "conv3", "ax_l1l3"):
            data[r["class"]][k].append(float(r[k]))
    for r in optics:
        tag = f"{float(r['wavelength_um']):.3f}_{float(r['m_re']):.2f}_{float(r['m_im']):.2f}"
        c = data[r["class"]]
        c[f"Qabs_{tag}"].append(float(r["Q_abs"]))
        c[f"Qsca_{tag}"].append(float(r["Q_sca"]))
        c[f"Qext_{tag}"].append(float(r["Q_ext"]))
        c[f"ReS_{tag}"].append(0.5 * (float(r["S1_re"]) + float(r["S2_re"])))  # polarisation mean
        c[f"ImS_{tag}"].append(-0.5 * (float(r["S1_im"]) + float(r["S2_im"])))  # |Im S(0)|, Im < 0
        c[f"conv_{tag}"].append(int(r["converged"]))
    return {k: {q: np.array(v) for q, v in d.items()} for k, d in data.items()}


def mse(v):
    return v.mean(), v.std(ddof=1) / np.sqrt(len(v))


def table(data):
    tag0 = f"{WL0:.3f}_{M0[0]:.2f}_{M0[1]:.2f}"
    cols = [("coord_005", "coord", "abs"), ("conv3", "conv3", "abs"), ("ax_l1l3", "ax13", "abs"),
            ("R_S_um", "R_S", "rel"), ("PA_um2", "PA", "rel"), ("d_mob_um", "d_mob", "rel"),
            (f"Qabs_{tag0}", "Q_abs", "rel"), (f"Qsca_{tag0}", "Q_sca", "rel"),
            (f"Qext_{tag0}", "Q_ext", "rel"), (f"ImS_{tag0}", "|Im S(0)|", "rel")]
    print(f"Optics at lambda = {WL0} um, m = {M0[0]} + {M0[1]}i; a = 15 nm, N = 150")
    print("Structure columns: class means.  Mobility and optics: class mean for R/flage/f, "
          "then % difference from it (z in brackets)")
    for f in FRACS:
        ref = data[f"R/flage/{f}"]
        print(f"\n--- target: R/flage/{f}  (Rg_F/a = {ref['rgf'].mean():.3f}) ---")
        print(f"{'class':18s}" + "".join(f"{lab:>17s}" for _, lab, _ in cols))
        names = [f"R/flage/{f}", f"R/tcca/{f}"] + [f"G/{m}/{d:.1f}/{f}" for m in ("tcca", "ptsa")
                                                  for d in DFS]
        for name in names:
            if name not in data:
                continue
            d = data[name]
            out = []
            for key, _, kind in cols:
                m, e = mse(d[key])
                if kind == "abs":
                    out.append(f"{m:.3f}")
                elif name == f"R/flage/{f}":
                    out.append(f"{m:.4g}")
                else:
                    mr, er = mse(ref[key])
                    diff = m - mr
                    out.append(f"{100 * diff / mr:+.1f}% ({diff / np.hypot(e, er):+.1f})")
            print(f"{name:18s}" + "".join(f"{o:>17s}" for o in out))
    for name in ("I/flage", "I/tcca"):
        d = data[name]
        print(f"\n{name}: Rg_F/a {d['rgf'].mean():.3f}, coord {d['coord_005'].mean():.3f}, "
              f"conv3 {d['conv3'].mean():.3f}, R_S {d['R_S_um'].mean():.4f} um, "
              f"PA {d['PA_um2'].mean():.5f} um2, d_mob {d['d_mob_um'].mean():.4f} um, "
              f"Q_abs {d[f'Qabs_{tag0}'].mean():.4f}, Q_sca {d[f'Qsca_{tag0}'].mean():.4f}")
    nconv = sum(int((1 - d[k]).sum()) for d in data.values() for k in d if k.startswith("conv_"))
    print(f"\nnon-converged MSTM solutions: {nconv}")


def figure(data, path):
    tag0 = f"{WL0:.3f}_{M0[0]:.2f}_{M0[1]:.2f}"
    qs = [("d_mob_um", r"$d_\mathrm{mob}$"), ("PA_um2", r"$\langle PA\rangle$"),
          ("R_S_um", r"$R_\mathrm{S}$"), (f"Qabs_{tag0}", r"$Q_\mathrm{abs}$"),
          (f"Qsca_{tag0}", r"$Q_\mathrm{sca}$"), ("coord_005", "coordination")]
    style = {"tcca": dict(color="#f47230", marker="s", ls="--"),
             "ptsa": dict(color="#7c4dff", marker="o", ls="-")}
    fig, ax = plt.subplots(len(FRACS), len(qs), figsize=(14, 7.2), constrained_layout=True)
    for i, f in enumerate(FRACS):
        ref = data[f"R/flage/{f}"]
        for j, (key, lab) in enumerate(qs):
            a = ax[i, j]
            m, e = mse(ref[key])
            norm = 1.0 if key == "coord_005" else m
            a.axhspan((m - 1.96 * e) / norm, (m + 1.96 * e) / norm, color="#257de0", alpha=0.25,
                      lw=0)
            a.axhline(m / norm, color="#257de0", lw=1.2, label="restructured (FLAGE)")
            for meth in ("tcca", "ptsa"):
                xs, ys, es = [], [], []
                for d in DFS:
                    name = f"G/{meth}/{d:.1f}/{f}"
                    if name in data:
                        mm, ee = mse(data[name][key])
                        xs.append(d)
                        ys.append(mm / norm)
                        es.append(1.96 * ee / norm)
                a.errorbar(xs, ys, yerr=es, capsize=2, lw=1.3, ms=4, label=meth.upper(),
                           **style[meth])
            a.set_xticks(DFS)
            if i == 0:
                a.set_title(lab + ("" if key == "coord_005" else " / restructured"), fontsize=9.5)
            if j == 0:
                a.set_ylabel(f"{f}% necks\n($R_{{g,F}}/a$ = {ref['rgf'].mean():.2f})", fontsize=9)
            if i == len(FRACS) - 1:
                a.set_xlabel(r"generator $D_\mathrm{f}$")
            a.grid(True, color="0.9", lw=0.6)
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)
    ax[0, 0].legend(frameon=False, fontsize=7.5)
    fig.savefig(path)
    print(f"saved {path}")


def main(optics_path, mobility_path, fig_path):
    data = collect(read_csv(optics_path), read_csv(mobility_path))
    table(data)
    figure(data, fig_path)


if __name__ == "__main__":
    main(*sys.argv[1:])
