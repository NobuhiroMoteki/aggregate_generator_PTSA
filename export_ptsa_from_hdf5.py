#!/usr/bin/env python3
"""
Export one aggregate from an HDF5 file to MSTM input format (.ptsa).

Usage examples
--------------
# List available HDF5 files:
python export_ptsa_from_hdf5.py --list

# Export a specific aggregate (all parameters are required):
python export_ptsa_from_hdf5.py \
    --h5_file aggregates_20260316_00.h5 \
    --mean_rp 0.020 --rel_std_rp 0.15 --k 0.90 \
    --Df 2.40 --Np 100 --agg_num 0

# Show all options:
python export_ptsa_from_hdf5.py -h
"""

import argparse
import glob
import os
import sys

import h5py


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def make_h5key(mean_rp, rel_std_rp, k, Df, Np, agg_num):
    """HDF5 group path: constants first, then sweep variables."""
    return f"{mean_rp:.4f}/{rel_std_rp:.2f}/{k:.3f}/{Df:.2f}/{Np:05d}/{agg_num}"


# ---------------------------------------------------------------------------
# List available files
# ---------------------------------------------------------------------------
def list_files(out_dir):
    """Print available HDF5 files and catalog CSVs."""
    h5_files = sorted(glob.glob(os.path.join(out_dir, "aggregates_????????_??.h5")))
    cat_files = sorted(glob.glob(os.path.join(out_dir, "catalog_????????_??.csv")))

    print("Available HDF5 files:")
    if h5_files:
        for f in h5_files:
            print(f"  {os.path.basename(f)}")
    else:
        print("  (none)")

    print("Corresponding catalogs:")
    if cat_files:
        for f in cat_files:
            print(f"  {os.path.basename(f)}")
    else:
        print("  (none)")


# ---------------------------------------------------------------------------
# Export function
# ---------------------------------------------------------------------------
def export_to_ptsa(h5_path, mean_rp, rel_std_rp, k, Df, Np, agg_num,
                   output_dir="./generated_agg_files/"):
    """
    Read one aggregate from HDF5 and write a .ptsa file for MSTM input.

    Parameters (ordered: constants first, then sweep variables)
    ----------
    h5_path      : str   path to the HDF5 file
    mean_rp      : float mean monomer radius [um]
    rel_std_rp   : float relative std of monomer radius [-]
    k            : float fractal prefactor
    Df           : float fractal dimension
    Np           : int   number of monomers
    agg_num      : int   aggregate index
    output_dir   : str   directory to write the .ptsa file

    Returns
    -------
    fpath : str   path of the written file
    """
    key = make_h5key(mean_rp, rel_std_rp, k, Df, Np, agg_num)

    with h5py.File(h5_path, "r") as h5f:
        if key not in h5f:
            # Collect a few full paths as examples
            examples = []
            def _collect(name, obj):
                if isinstance(obj, h5py.Group) and "xp" in obj:
                    examples.append(name)
            h5f.visititems(_collect)

            msg = (
                f"Aggregate not found in HDF5.\n"
                f"  Requested key : {key}\n"
                f"  h5_path       : {h5_path}\n"
                f"\n"
                f"  Key format: {{mean_rp:.4f}}/{{rel_std_rp:.2f}}/{{k:.3f}}/{{Df:.2f}}/{{Np:05d}}/{{agg_num}}\n"
                f"  Parameter order: mean_rp / rel_std_rp / k / Df / Np / agg_num\n"
            )
            if examples:
                msg += f"\n  Example keys in this file (first 5):\n"
                for ex in examples[:5]:
                    msg += f"    {ex}\n"
            msg += "\n  Hint: check the corresponding catalog CSV for available entries."
            raise KeyError(msg)

        grp = h5f[key]
        xp = grp["xp"][:]   # shape (Np, 3)  [um]
        rp = grp["rp"][:]   # shape (Np,)    [um]

    Np_act = xp.shape[0]

    # Filename: constants first, then sweep variables
    ofname = (
        f"meanRp={mean_rp:.3f}um_rstdRp={rel_std_rp:.2f}_k={k:.3f}"
        f"_Df={Df:.2f}_Np={Np_act:05d}_agg_num={agg_num:01d}.ptsa"
    )
    fpath = os.path.join(output_dir, ofname)

    os.makedirs(output_dir, exist_ok=True)
    with open(fpath, "w") as f:
        for ip in range(Np_act):
            f.write("{:13.7e}  {:13.7e}  {:13.7e}  {:13.7e}\n".format(
                xp[ip, 0], xp[ip, 1], xp[ip, 2], rp[ip]))

    print(f"Exported: {fpath}")
    return fpath


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
REQUIRED_PARAMS = [
    ("--h5_file",    "h5_file",    str,   "HDF5 filename (e.g. aggregates_20260316_00.h5)"),
    ("--mean_rp",    "mean_rp",    float, "Mean monomer radius [um]"),
    ("--rel_std_rp", "rel_std_rp", float, "Relative std of monomer radius"),
    ("--k",          "k",          float, "Fractal prefactor"),
    ("--Df",         "Df",         float, "Fractal dimension"),
    ("--Np",         "Np",         int,   "Number of monomers"),
    ("--agg_num",    "agg_num",    int,   "Aggregate index"),
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Export one aggregate from HDF5 to MSTM input format (.ptsa).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--list", action="store_true",
                   help="List available HDF5 files and catalogs, then exit")

    # All parameters are required (no defaults) to avoid silent mismatches
    for flag, dest, typ, hlp in REQUIRED_PARAMS:
        p.add_argument(flag, dest=dest, type=typ, default=None, help=f"{hlp} (REQUIRED)")

    p.add_argument("--out_dir", type=str, default="./generated_agg_files/",
                   help="Output directory (default: ./generated_agg_files/)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.list:
        list_files(args.out_dir)
        raise SystemExit(0)

    # Validate: all parameters must be explicitly provided
    missing = [flag for flag, dest, _, _ in REQUIRED_PARAMS if getattr(args, dest) is None]
    if missing:
        print(f"Error: the following arguments are required: {', '.join(missing)}", file=sys.stderr)
        print(f"", file=sys.stderr)
        print(f"  Required parameter order:", file=sys.stderr)
        print(f"    --h5_file  --mean_rp  --rel_std_rp  --k  --Df  --Np  --agg_num", file=sys.stderr)
        print(f"", file=sys.stderr)
        print(f"  Example:", file=sys.stderr)
        print(f"    python export_ptsa_from_hdf5.py \\", file=sys.stderr)
        print(f"        --h5_file aggregates_20260316_00.h5 \\", file=sys.stderr)
        print(f"        --mean_rp 0.020 --rel_std_rp 0.15 --k 0.90 \\", file=sys.stderr)
        print(f"        --Df 2.40 --Np 100 --agg_num 0", file=sys.stderr)
        print(f"", file=sys.stderr)
        print(f"  Run with --list to see available HDF5 files, or -h for help.", file=sys.stderr)
        raise SystemExit(1)

    h5_path = os.path.join(args.out_dir, args.h5_file)

    export_to_ptsa(
        h5_path=h5_path,
        mean_rp=args.mean_rp,
        rel_std_rp=args.rel_std_rp,
        k=args.k,
        Df=args.Df,
        Np=args.Np,
        agg_num=args.agg_num,
        output_dir=args.out_dir,
    )
