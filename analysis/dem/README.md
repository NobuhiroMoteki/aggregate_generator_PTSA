# DEM restructuring runs with soot-dem (stage 1)

These scripts reproduce the restructuring simulations of Demidov, Gor and Khalizov,
*Phys. Rev. E* **110**, 054902 (2024), Fig. 8. They also repeat the runs with TCCA aggregates
from this repository. The results and their discussion are in
[docs/restructuring_review.pdf](../../docs/restructuring_review.pdf), Sec. "Reproduction".
The shape descriptors (step 1) and the comparison of optics and mobility with generator aggregates
(step 2) are in Sec. "Shape, optics and mobility of the restructured aggregates" of the same note.

| File | Purpose |
| --- | --- |
| `prepare_inputs.py` | builds a work directory from the Supplemental Material archive (FLAGE or TCCA aggregates) |
| `run_one.sh` | runs one simulation on one thread and appends its exit code to `done.log` |
| `run_all.sh` | runs all 500 jobs of a work directory with `xargs -P` |
| `dem_metrics.py` | coordination number and convexity per dump → `.npz` |
| `dem_summary.py` | tables, TCCA − FLAGE differences and the figure `docs/figs/dem_compare.pdf` |
| `metrics_flage.npz`, `metrics_tcca.npz` | metrics of the 2 × 500 runs of 2026-11-26 to 2026-11-28 |
| `dem_shape.py`, `dem_shape_summary.py` | shape descriptors (Rg, gyration tensor, projected area, S(q)) and coordinates of the first and last dumps |
| `shape_flage.npz`, `shape_tcca.npz` | their output for the 2 × 500 runs |
| `step2_generate.py`, `step2b_generate.py` | restructured aggregates and generator aggregates (TCCA, PTSA) matched in N and Rg_F, or with smaller Rg_F for matching d_mob |
| `step2_mstm.jl`, `step2_order_check.jl` | optics with MSTMforCAS.jl and its convergence in the truncation order |
| `step2_mobility.py` | mobility (external module, see its docstring) and structure descriptors |
| `step2_harmonic.py` | bound on the harmonic orientational mean of the free molecular friction (ellipsoid; external module) |
| `step2_pa_tensor.py` | check of the ellipsoid f_A/f_H against a projected-area tensor |
| `step2_summary.py`, `step2b_summary.py` | comparison at equal Rg_F and at equal d_mob |
| `step2*_aggregates.h5`, `step2*_optics.csv`, `step2*_mobility.csv`, `step2_order_check.csv` | inputs and results of step 2 |

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

## Comparison with generator aggregates (optics and mobility)

The restructured aggregates are compared with TCCA and PTSA aggregates of the same N = 150,
first at equal Rg_F and then at equal mobility diameter. Monomer radius 15 nm; wavelengths
0.453, 0.638 and 0.834 µm; refractive indices 1.75 + 0.60i and 1.95 + 0.79i.

```bash
uv run python analysis/dem/step2_generate.py analysis/dem/step2_aggregates.h5 50
uv run python analysis/dem/step2b_generate.py analysis/dem/step2b_aggregates.h5 30
# on the compute host (MSTMforCAS.jl v1.2.1; mobility needs numpy, scipy, h5py)
julia -t 20 --project=PATH/TO/MSTMforCAS.jl analysis/dem/step2_mstm.jl step2_aggregates.h5 step2_optics.csv
python analysis/dem/step2_mobility.py step2_aggregates.h5 MOBILITY_MODULE.py step2_mobility.csv 20
# ellipsoid bound on the harmonic orientational mean (adds d_mob_H_um); check: step2_pa_tensor.py
python analysis/dem/step2_harmonic.py step2_aggregates.h5 step2_mobility.csv ANISOTROPY_MODULE.py \
    MOBILITY_MODULE.py step2_mobility.csv
# (same commands for step2b_aggregates.h5)
python analysis/dem/step2_summary.py analysis/dem/step2_optics.csv analysis/dem/step2_mobility.csv docs/figs/dem_step2.pdf
python analysis/dem/step2b_summary.py analysis/dem/step2_optics.csv analysis/dem/step2_mobility.csv \
    analysis/dem/step2b_optics.csv analysis/dem/step2b_mobility.csv   # add --harmonic for the bound
```

A DMA measures the mean of the mobility, i.e. the harmonic orientational mean of the friction.
R_S corresponds to it, but the adjusted-sphere expression with ⟨PA⟩ uses the arithmetic mean of
the free molecular friction. The main results use this expression, d_mob(A), as is usual for
aggregates. `step2_harmonic.py` adds d_mob(H) with PA replaced by PA·f_H/f_A, where f_A/f_H comes
from the ellipsoid of the same gyration tensor. The ellipsoid treats the aggregate as a solid
convex body and overestimates f_A/f_H − 1 of open aggregates: 12–19 % against 2–6 % from a
projected-area tensor that includes the shielding between monomers (`step2_pa_tensor.py`).
d_mob(H) is therefore a bound on the effect (−3.7 to −5.2 % for the restructured aggregates),
not a correction. The exact value needs the free molecular friction tensor of the aggregate.

Result (d_mob(A)): at necking fractions up to 50 % no generator class reproduces the
restructured aggregates. Matched in Rg_F or in d_mob, the generator aggregates absorb 7–12 % less,
because their coordination number stays near 2. With all necks intact, generator aggregates
matched in d_mob agree within about 3 % in all quantities, except Q_sca at 0.834 µm (up to 8 %).
With the ellipsoid bound d_mob(H) the conclusion for necking fractions up to 50 % is unchanged
(absorption −6 to −10 %, scattering up to +28 %); with all necks intact only TCCA (Df = 2.2)
stays within 3.5 %, except Q_sca (6.4 %).
