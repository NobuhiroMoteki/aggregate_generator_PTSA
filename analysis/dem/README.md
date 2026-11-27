# DEM restructuring runs with soot-dem (stage 1)

These scripts reproduce the restructuring simulations of Demidov, Gor and Khalizov,
*Phys. Rev. E* **110**, 054902 (2024), Fig. 8. They also repeat the runs with TCCA aggregates
from this repository. The results and their discussion are in
[docs/restructuring_review.pdf](../../docs/restructuring_review.pdf), Sec. "Reproduction".

| File | Purpose |
| --- | --- |
| `prepare_inputs.py` | builds a work directory from the Supplemental Material archive (FLAGE or TCCA aggregates) |
| `run_one.sh` | runs one simulation on one thread and appends its exit code to `done.log` |
| `run_all.sh` | runs all 500 jobs of a work directory with `xargs -P` |
| `dem_metrics.py` | coordination number and convexity per dump → `.npz` |
| `dem_summary.py` | tables, TCCA − FLAGE differences and the figure `docs/figs/dem_compare.pdf` |
| `metrics_flage.npz`, `metrics_tcca.npz` | metrics of the 2 × 500 runs of 2026-11-26 to 2026-11-28 |

## Prerequisites

- **soot-dem** (MIT licence), tag `v1` (commit `7572f45`), which corresponds to the submitted article.
  It is not included in this repository. Build only the `restructuring` target:

  ```bash
  git clone --branch v1 --recurse-submodules --shallow-submodules \
      https://github.com/egor-demidov/soot-dem.git soot-dem
  cmake -S soot-dem -B soot-dem/build -DCMAKE_BUILD_TYPE=Release
  cmake --build soot-dem/build --target restructuring -j 8
  ```

  We built it with g++ 11.4 and CMake 3.22 on Ubuntu 22.04 (WSL2) and ran the executable on the A6000 desktop.
- **Input archive** `dem-input-files.tar.gz` from the Supplemental Material of the paper.
  It is not redistributed here. Place it in `docs/literature/`, which git ignores.

## Procedure

```bash
# 1. work directories (the TCCA variant generates the 50 aggregates, seeds SeedSequence(2026112600 + i))
uv run python analysis/dem/prepare_inputs.py --archive docs/literature/dem-input-files.tar.gz \
    --out ~/dem_flage --aggregates flage --binary soot-dem/build/restructuring
uv run python analysis/dem/prepare_inputs.py --archive docs/literature/dem-input-files.tar.gz \
    --out ~/dem_tcca --aggregates tcca --binary soot-dem/build/restructuring

# 2. runs (on the compute host; detached so that the session can close)
nohup setsid analysis/dem/run_all.sh ~/dem_flage 10 > ~/dem_flage/run_all.log 2>&1 < /dev/null &

# 3. metrics (needs numpy and scipy) and summary
python analysis/dem/dem_metrics.py ~/dem_flage analysis/dem/metrics_flage.npz
python analysis/dem/dem_metrics.py ~/dem_tcca analysis/dem/metrics_tcca.npz
python analysis/dem/dem_summary.py docs/figs/dem_compare.pdf \
    analysis/dem/metrics_flage.npz analysis/dem/metrics_tcca.npz
```

`prepare_inputs.py` changes one setting of the published inputs. It reduces the number of
dumps from 500 to 50 (one per ns) to save disk space. All physical parameters stay as published.
Run 2 needs the repository scripts on the compute host, or a copy of `run_one.sh` and `run_all.sh`.

## Cost

One run (N = 150, 10^7 steps of 5 fs) takes about 21 min on one core.
500 runs with 10 parallel jobs took about 18 h on the desktop. The 50 dumps of one run take
about 0.8 MB, so a work directory needs about 0.5 GB.

The runs are chaotic. Results change with the compiler and the number of OpenMP threads, so
only statistics over the 50 aggregates can be compared. `run_one.sh` uses one thread per run.
