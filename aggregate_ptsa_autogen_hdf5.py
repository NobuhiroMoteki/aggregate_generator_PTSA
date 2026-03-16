#!/usr/bin/env python3
"""
Batch generator for fractal-like aggregates using the PTSA algorithm.
Outputs are stored in a single HDF5 file with a companion catalog CSV.

Usage examples
--------------
# Run with default parameters (same as the notebook defaults):
python aggregate_ptsa_autogen_hdf5.py

# Custom sweep:
python aggregate_ptsa_autogen_hdf5.py \
    --mean_rp 0.020 --Np_min 100 --Np_max 1000 --num_Np 10 \
    --Df_min 2.40 --Df_max 2.80 --num_Df 5 --agg_num 0 1 2

# Show all options:
python aggregate_ptsa_autogen_hdf5.py -h
"""

import argparse
import glob
import os
import time
from datetime import date

import h5py
import numba
import numpy as np
import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------------
# PTSA aggregate generator (Numba JIT)
# ---------------------------------------------------------------------------
@numba.njit
def ptsa(Np, mean_rp, rel_std_rp, Df, k, max_search_num, rng):
    """
    Polydisperse tunable sequential aggregation (PTSA) method for generating
    fractal-like aggregates.  Monomer radius follows a normal distribution
    with specified mean and standard deviation.

    Reference: Singh & Tsotsas 2022, https://doi.org/10.1016/j.ces.2021.117022
    """

    tol = 0.001 + (2.95 - Df) * 0.0033

    # Box-Muller for standard normal variates
    u1 = rng.random(size=Np)
    u2 = rng.random(size=Np)
    r = np.sqrt(-2.0 * np.log(u1))
    theta = 2.0 * np.pi * u2
    snv = r * np.cos(theta)

    # Clip outliers beyond 2 sigma
    for i, val in enumerate(snv):
        if val < -2:
            snv[i] = -2
        elif val > 2:
            snv[i] = 2

    rp = mean_rp * rel_std_rp * snv + mean_rp
    density = 1.0
    mp = (4 * np.pi / 3) * density * rp ** 3
    mp3t = np.broadcast_to(mp, (3, Np))
    mp3 = mp3t.transpose()
    xp = np.zeros((Np, 3))
    minimum_separation = mean_rp * 1e-6

    xp[0, 0], xp[0, 1], xp[0, 2] = 0.0, 0.0, 0.0
    phi = rng.random() * 2 * np.pi
    u = rng.random() * 2 - 1.0
    sqrt1mu2 = np.sqrt(1 - u ** 2)
    ux, uy, uz = sqrt1mu2 * np.cos(phi), sqrt1mu2 * np.sin(phi), u
    R = rp[0] + rp[1] + minimum_separation
    xp[1, 0], xp[1, 1], xp[1, 2] = R * ux, R * uy, R * uz
    xp[0:2, :] = xp[0:2, :] - np.sum(mp3[0:2, :] * xp[0:2, :], axis=0) / np.sum(mp[0:2])

    Np_final = 0
    for ip in range(2, Np):
        n = ip + 1
        Rpn = rp[ip]
        if n <= k ** (3 / (3 - Df)):
            Df_tuned = 3
        else:
            Df_tuned = Df * (np.log(Np) / np.log(Np / k))
        R2 = (n ** 2 * Rpn ** 2 / (n - 1)) * n ** (2 / Df_tuned) \
             - (n * Rpn ** 2 / (n - 1)) - n * Rpn ** 2 * (n - 1) ** (2 / Df_tuned)
        R = np.sqrt(R2)

        maximum_separation = mean_rp * tol * (1 + 5 * (ip / Np) * (2.95 - Df) / 0.6)

        attached_cond = False
        for i_search in range(max_search_num):
            phi = rng.random() * 2 * np.pi
            u = rng.random() * 2 - 1.0
            sqrt1mu2 = np.sqrt(1 - u * u)
            x0 = R * sqrt1mu2 * np.cos(phi)
            x1 = R * sqrt1mu2 * np.sin(phi)
            x2 = R * u

            overlapping = False
            proximate = False
            for j in range(ip):
                dx = xp[j, 0] - x0
                dy = xp[j, 1] - x1
                dz = xp[j, 2] - x2
                surf_dist = np.sqrt(dx * dx + dy * dy + dz * dz) - (rp[j] + Rpn)
                if surf_dist < minimum_separation:
                    overlapping = True
                    break
                if surf_dist < maximum_separation:
                    proximate = True

            if overlapping:
                continue

            if proximate:
                attached_cond = True
                xp[ip, 0] = x0
                xp[ip, 1] = x1
                xp[ip, 2] = x2
                xp[0:n, :] = xp[0:n, :] - np.sum(mp3[0:n, :] * xp[0:n, :], axis=0) / np.sum(mp[0:n])
                break

        Np_final = n
        if not attached_cond:
            break

    if Np_final == Np:
        Rg = np.sqrt((1 / np.sum(mp[:])) * np.sum(
            mp3[:, 0] * ((xp[:, 0] ** 2 + xp[:, 1] ** 2 + xp[:, 2] ** 2) + 3 / 5 * rp[:] ** 2)))
        Re = np.sqrt(5 / 3) * Rg
        Vagg = 4 * np.pi / 3 * Re ** 3
        V = np.sum(mp[:]) / density
        Rve = np.cbrt(3 * V / (4 * np.pi))
        eps_agg = 1 - V / Vagg
    else:
        Rg = 0
        Re = 0
        Vagg = 0
        V = 0
        Rve = 0
        eps_agg = 0

    return Np_final, xp, rp, V, Rve, Rg, eps_agg


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def make_h5key(mean_rp, rel_std_rp, k, Df, Np, agg_num):
    """HDF5 group path: constants first, then sweep variables."""
    return f"{mean_rp:.4f}/{rel_std_rp:.2f}/{k:.3f}/{Df:.2f}/{Np:05d}/{agg_num}"


# ---------------------------------------------------------------------------
# JIT warm-up
# ---------------------------------------------------------------------------
def warmup():
    """Run ptsa once with small Np to trigger Numba compilation."""
    print("JIT compiling ptsa ... ", end="", flush=True)
    t0 = time.perf_counter()
    rng = np.random.default_rng()
    ptsa(20, 0.015, 0.1, 2.9, 0.9, 1000000, rng)
    print(f"done ({time.perf_counter() - t0:.1f}s)")


# ---------------------------------------------------------------------------
# Main batch loop
# ---------------------------------------------------------------------------
def run_batch(args):
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Determine output filenames
    date_str = date.today().strftime("%Y%m%d")
    existing = sorted(glob.glob(os.path.join(out_dir, f"aggregates_{date_str}_??.h5")))
    next_id = len(existing)
    h5_fname = f"aggregates_{date_str}_{next_id:02d}.h5"
    catalog_fname = f"catalog_{date_str}_{next_id:02d}.csv"
    h5_path = os.path.join(out_dir, h5_fname)
    catalog_path = os.path.join(out_dir, catalog_fname)
    print(f"HDF5 output : {h5_path}")
    print(f"Catalog     : {catalog_path}")

    Df_arr = np.linspace(args.Df_min, args.Df_max, args.num_Df)
    Np_arr = np.linspace(args.Np_min, args.Np_max, args.num_Np).astype(int)

    n_agg = len(args.agg_num)
    total = n_agg * len(Df_arr) * len(Np_arr)
    Rve_min = args.mean_rp * args.Np_min ** (1 / 3)
    Rve_max = args.mean_rp * args.Np_max ** (1 / 3)
    print(f"Parameter grid: {n_agg} agg_num x {len(Df_arr)} Df x {len(Np_arr)} Np = {total} aggregates")
    print(f"Estimated Rve range: {Rve_min:.4f} -- {Rve_max:.4f} um "
          f"(monodisperse approx: Rve = mean_rp * Np^(1/3))\n")

    new_records = []

    # Build flat task list for tqdm
    tasks = [(agg_num, Df, Np)
             for agg_num in args.agg_num
             for Df in Df_arr
             for Np in Np_arr]

    with h5py.File(h5_path, "a") as h5f:
        pbar = tqdm(tasks, desc="Generating", unit="agg",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                               "[{elapsed}<{remaining}, {rate_fmt}]")
        for agg_num, Df, Np in pbar:
            key = make_h5key(args.mean_rp, args.rel_std_rp, args.k, Df, Np, agg_num)
            pbar.set_postfix_str(f"Df={Df:.2f} Np={Np:5d} agg={agg_num}")

            if key in h5f:
                continue

            for i_try in range(args.max_try):
                tqdm.write(f"  agg_num={agg_num:02d}, Df={Df:.2f}, Np={Np:05d}, "
                           f"try={i_try:02d}")
                rng = np.random.default_rng()
                t0 = time.perf_counter()
                Np_final, xp, rp, V, Rve, Rg, eps_agg = ptsa(
                    Np, args.mean_rp, args.rel_std_rp, Df, args.k,
                    args.max_search, rng)
                elapsed_min = (time.perf_counter() - t0) / 60
                tqdm.write(f"    Np_final={Np_final:05d}, {elapsed_min:.2f} min")

                if Np_final != Np:
                    if i_try < args.max_try - 1:
                        continue
                    tqdm.write(f"    FAILED after {args.max_try} tries: {key}")
                    break

                # Success: write to HDF5
                grp = h5f.create_group(key)
                grp.create_dataset("xp", data=xp,
                                   compression="gzip", compression_opts=4)
                grp.create_dataset("rp", data=rp,
                                   compression="gzip", compression_opts=4)
                grp.attrs.update({
                    "mean_rp":    float(args.mean_rp),
                    "rel_std_rp": float(args.rel_std_rp),
                    "k":          float(args.k),
                    "Df":         float(round(Df, 2)),
                    "Np":         int(Np),
                    "agg_num":    int(agg_num),
                    "Rve":        float(Rve),
                    "Rg":         float(Rg),
                    "eps_agg":    float(eps_agg),
                })
                new_records.append({
                    "mean_rp":    float(args.mean_rp),
                    "rel_std_rp": float(args.rel_std_rp),
                    "k":          float(args.k),
                    "Df":         float(round(Df, 2)),
                    "Np":         int(Np),
                    "agg_num":    int(agg_num),
                    "Rve":        float(Rve),
                    "Rg":         float(Rg),
                    "eps_agg":    float(eps_agg),
                    "h5_file":    h5_fname,
                    "h5_key":     key,
                })
                break

        pbar.close()

    # Write catalog CSV
    if new_records:
        df_catalog = pd.DataFrame(new_records)
        df_catalog.to_csv(catalog_path, index=False)
        print(f"Catalog saved: {catalog_path} ({len(new_records)} entries).")
    else:
        print("No new aggregates were generated.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Batch-generate PTSA fractal aggregates and store in HDF5.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Constant parameters
    p.add_argument("--mean_rp",    type=float, default=0.015,
                   help="Mean monomer radius [um] (default: 0.015)")
    p.add_argument("--rel_std_rp", type=float, default=0.10,
                   help="Relative std of monomer radius (default: 0.10)")
    p.add_argument("--k",          type=float, default=0.95,
                   help="Fractal prefactor (default: 0.95)")

    # Sweep: Df
    p.add_argument("--Df_min",  type=float, default=2.35,
                   help="Min fractal dimension (default: 2.35)")
    p.add_argument("--Df_max",  type=float, default=2.95,
                   help="Max fractal dimension (default: 2.95)")
    p.add_argument("--num_Df",  type=int,   default=13,
                   help="Number of Df grid points (default: 13)")

    # Sweep: Np
    p.add_argument("--Np_min",  type=int, default=350,
                   help="Min number of monomers (default: 350)")
    p.add_argument("--Np_max",  type=int, default=2800,
                   help="Max number of monomers (default: 2800)")
    p.add_argument("--num_Np",  type=int, default=50,
                   help="Number of Np grid points (default: 50)")

    # Sweep: agg_num
    p.add_argument("--agg_num", type=int, nargs="+", default=list(range(10)),
                   help="List of aggregate indices (default: 0 1 2 ... 9)")

    # Solver limits
    p.add_argument("--max_search", type=int, default=100_000_000,
                   help="Max iterations per monomer search (default: 1e8)")
    p.add_argument("--max_try",    type=int, default=20,
                   help="Max retries per aggregate (default: 20)")

    # Output
    p.add_argument("--out_dir", type=str, default="./generated_agg_files/",
                   help="Output directory (default: ./generated_agg_files/)")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    warmup()
    run_batch(args)
