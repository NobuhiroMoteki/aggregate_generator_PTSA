#!/usr/bin/env python3
"""
Prepare a soot-dem work directory for the restructuring runs of Demidov et al. (2024).

The input files are taken from the Supplemental Material archive of the paper
(dem-input-files.tar.gz, not redistributed here).  Two variants are supported:

  --aggregates flage : the 50 published FLAGE aggregates (reproduction of Fig. 8)
  --aggregates tcca  : 50 monodisperse TCCA aggregates from this repository with the
                       same Df = 1.8, kf = 1.4 and N = 150, written in FLAGE format

In both variants the number of dumps is reduced from 500 to 50 (one per ns) to save
disk space.  All other parameters are left as published.

The work directory receives
  bin/restructuring               copy of the soot-dem executable (if --binary is given)
  aggregate_bank/ or aggregate_bank_tcca/
  restructuring_simulations/      input tree of the archive
  jobs.txt                        one run directory per line (500 lines)

Usage (from the repository root):
    python analysis/dem/prepare_inputs.py --archive docs/literature/dem-input-files.tar.gz \
        --out WORK_DIR --aggregates {flage,tcca} [--binary PATH/TO/restructuring]
"""

import argparse
import json
import pathlib
import re
import shutil
import sys
import tarfile
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import tcca  # noqa: E402

N_AGG = 50
FRACS = [0, 15, 30, 50, 70, 80, 90, 95, 98, 100]
N_DUMPS = 50
SEED_BASE = 20261126 * 100  # aggregate i uses np.random.SeedSequence(SEED_BASE + i)


def write_flage_xml(path, xp, comment):
    """Write monomer centres xp (shape: (N, 3), units of the monomer radius) as a FLAGE file."""
    lines = ['<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
             '<geometry_complex version="1.00">',
             f'  <data_particle algorithm="{comment}">']
    for k, p in enumerate(xp):
        lines += [f'    <particle ID="{k}" type="sphere">',
                  '      <radius> 1.0 </radius>',
                  '      <position> %.17g , %.17g , %.17g </position>' % (p[0], p[1], p[2]),
                  '      <refraction> 1 </refraction>',
                  '    </particle>']
    lines += ['  </data_particle>', '</geometry_complex>']
    path.write_text("\n".join(lines) + "\n")


def make_tcca_bank(out_dir, Np=150, Df=1.8, kf=1.4):
    """Generate N_AGG monodisperse TCCA aggregates (unit monomer radius) in FLAGE format."""
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = []
    for i in range(N_AGG):
        seed = np.random.SeedSequence(SEED_BASE + i)
        r = tcca.tcca(Np, 1.0, 0.0, Df, kf, rng=np.random.default_rng(seed))
        if r[0] != Np:
            raise RuntimeError(f"TCCA failed for aggregate {i}")
        xp = r[1]  # shape: (Np, 3)
        if not tcca.check_geometry(xp, r[2], 1e-2)["ok"]:
            raise RuntimeError(f"TCCA geometry check failed for aggregate {i}")
        comment = (f"tcca (aggregate_generator_PTSA v0.5.0), Df={Df}, kf={kf}, N={Np}, "
                   f"seed_entropy={seed.entropy}")
        write_flage_xml(out_dir / f"aggregate_{i}.xml", xp, comment)
        rg = float(np.sqrt(((xp - xp.mean(0)) ** 2).sum(1).mean()))
        meta.append(dict(i=i, seed_entropy=str(seed.entropy), Rg=rg))
    (out_dir / "seeds.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"TCCA bank: {N_AGG} aggregates, mean Rg = {np.mean([m['Rg'] for m in meta]):.3f} r")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--archive", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--aggregates", choices=("flage", "tcca"), required=True)
    ap.add_argument("--binary", type=pathlib.Path, default=None)
    a = ap.parse_args()

    out = a.out.resolve()
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} exists and is not empty")
    out.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(a.archive) as tf:
            members = [m for m in tf.getmembers()
                       if m.name.startswith(("dem-input-files/aggregate_bank/",
                                             "dem-input-files/restructuring_simulations/"))]
            tf.extractall(tmp, members=members, filter="data")
        src = pathlib.Path(tmp) / "dem-input-files"
        shutil.copytree(src / "restructuring_simulations", out / "restructuring_simulations")
        if a.aggregates == "flage":
            shutil.copytree(src / "aggregate_bank", out / "aggregate_bank")

    common = out / "restructuring_simulations" / "common_all_simulations.xml"
    txt, n = re.subn(r'(<let id="n_dumps" type="integer">)\d+(</let>)',
                     rf"\g<1>{N_DUMPS}\g<2>", common.read_text())
    if n != 1:
        sys.exit("n_dumps not found in common_all_simulations.xml")
    common.write_text(txt)

    if a.aggregates == "tcca":
        make_tcca_bank(out / "aggregate_bank_tcca" / "size_150")
        for i in range(N_AGG):
            f = (out / "restructuring_simulations" / "size_150" / f"aggregate_{i}"
                 / "thin_coating" / "common_thin_coating_aggregate.xml")
            txt, n = re.subn(r"aggregate_bank/size_150/aggregate_\d+\.xml",
                             f"aggregate_bank_tcca/size_150/aggregate_{i}.xml", f.read_text())
            if n != 1:
                sys.exit(f"aggregate_path not found in {f}")
            f.write_text(txt)

    jobs = [f"restructuring_simulations/size_150/aggregate_{i}/thin_coating/frac_{f}"
            for i in range(N_AGG) for f in FRACS]
    for j in jobs:
        if not (out / j / "restructuring_thin_coating_input.xml").is_file():
            sys.exit(f"missing input in {j}")
    (out / "jobs.txt").write_text("\n".join(jobs) + "\n")

    if a.binary is not None:
        (out / "bin").mkdir()
        shutil.copy2(a.binary, out / "bin" / "restructuring")
    print(f"prepared {out} ({a.aggregates}, {len(jobs)} runs)")


if __name__ == "__main__":
    main()
