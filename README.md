# aggregate_generator_PTSA

[![Latest version](https://img.shields.io/github/v/tag/NobuhiroMoteki/aggregate_generator_PTSA?label=version&sort=semver)](https://github.com/NobuhiroMoteki/aggregate_generator_PTSA/tags)

## 📌 Description

A Python code for the polydisperse tunable sequential aggregation (PTSA) algorithm. Monomer positions are sequentially determined by a particle-cluster stochastic aggregation algorithm so as to satisfy the fractal scaling law, surface-attachment condition, and non-overlapping condition. The monomer position is arbitrary, not being restricted to a gridded lattice space. A notable advantage of this PTSA algorithm is its capability of generating compact fractal-like aggregates with high Df (up to 2.95), where the conventional cluster-cluster aggregation (CCA) algorithm will easily break down.

As an option, the same pipeline can generate aggregates by **tunable cluster-cluster aggregation (CCA)** (Filippov et al., 2000), translated to Python/Numba from the C++ code [nmoteki/aggregate_generator](https://github.com/nmoteki/aggregate_generator). PTSA and CCA give aggregates with the same (k, Df) and the same radius of gyration, but their internal geometry differs (see [Choosing the algorithm](#choosing-the-algorithm---method)). Generating both covers both possibilities.

The inner search loop is accelerated by JIT compilation via [Numba](https://numba.pydata.org/) and an early-exit scalar overlap check, achieving a **2.8–3.5× speedup** over a naive vectorized implementation.

### Main Features

- **`ptsa`**: aggregate generator with JIT compilation (Numba) and early-exit scalar overlap check
- **`tcca`** (`tcca.py`): tunable cluster-cluster aggregation, selected by `--method cca`; same inputs and outputs as `ptsa` plus CCA-specific options
- **Multi-core parallelism**: batch script distributes `agg_num` jobs across CPU cores via `ProcessPoolExecutor` (`--workers` option)
- Aggregate generator function (ptsa) with inputs and outputs defined as follows:

```text
def ptsa(Np, mean_rp, rel_std_rp, Df, k, max_search_num, rng)
    Polydisperse tunable sequential aggregation (PTSA) method for generating fractal-like aggregate.
    Monomer radius follows a normal distribution with specified mean and standard deviation.

    ==== Input parameters ===
    Np: number of monomers
    mean_rp: mean of the normal distribution of the monomer radius
    rel_std_rp: relative standard deviation of the normal distrituion of the monomer radius
    Df: fractal dimension
    k: fractal prefactor
    max_search_num: maximum number of iteration for searching a location of each monomer attached onto the surface of an aggregate (default: 100000000)
    rng: random number generator constructed by the numpy.random.default_rng()

    === Output variables ===
    Np_final: number of monomers in the generated aggregate. If the aggregate generation failed, Np_final < Np.
    xp: (x,y,z) coordinate of the center of individual monomers relative to the centroid of the aggregate. 2D of shape= (Np,3).
    rp: radii of individual monomers. 1D array of size= Np.
    V: volume of the aggregate.
    Rve: volume-equivalent radius of the aggregate.
    Rg: gyration radius of aggregate.
    eps_agg: porosity of the aggregate calculated using the gyration method.
```

### Limitations

The aggregate generation might fail depending on the combination of (k, Df). N.Moteki tested only k=0.95, and Df= 2.35~2.95.

CCA covers a narrower range: roughly k + Df < 3.5. At k = 0.95 it succeeds for Df = 1.8–2.6 and fails above about Df = 2.7 (see [Choosing the algorithm](#choosing-the-algorithm---method)).

---

## 🚀 Installation

The author developed and tested current aggregate_generator_PTSA using Python 3.13.12 in Windows 11 and WSL2 (Ubuntu on Windows 11) machines.

#### 1. Clone the repository

```sh
git clone https://github.com/NobuhiroMoteki/aggregate_generator_PTSA.git
cd aggregate_generator_PTSA
```

#### 2. Install dependencies

```sh
pip install -r requirements.txt
```

---

## 🔧 Usage

### Parameter ordering convention

Throughout the codebase, parameters are ordered as **constants first, then sweep variables**:

1. `mean_rp` — mean monomer radius [μm] (constant)
2. `rel_std_rp` — relative std of monomer radius (constant)
3. `k` — fractal prefactor (constant)
4. `Df` — fractal dimension (sweep variable)
5. `Np` — number of monomers (sweep variable)
6. `agg_num` — aggregate index (sweep variable, batch only)

This ordering is applied consistently to HDF5 group keys, HDF5 attributes, catalog CSV columns, and output filenames.

---

### Single execution

1. Open `aggregate_ptsa_single.ipynb` and specify the following parameters in the 4th cell:
   - mean monomer radius [μm]: `mean_rp`
   - relative standard deviation of monomer radius: `rel_std_rp`
   - fractal prefactor: `k`
   - fractal dimension: `Df`
   - number of monomers: `Np`
2. Execute all cells. A 3D scatter plot of the generated aggregate will appear and be saved as a 300 dpi JPEG.
3. The output `.ptsa` and `.jpg` files are written to `./generated_agg_files/` with the filename:

```text
meanRp={mean_rp:.3f}um_rstdRp={rel_std_rp:.2f}_k={k:.3f}_Df={Df:.2f}_Np={Np:05d}_Rve={Rve:.3f}um_Rg={Rg:.3f}um_epsagg={eps_agg:.3f}.ptsa
```

---

### Batch execution (parameter sweep) — HDF5 output

Generate a large number of aggregates over a parameter grid and store them efficiently in a single HDF5 file.

All parameters can be specified via command-line arguments:

```sh
# Run with default parameters:
python aggregate_ptsa_autogen_hdf5.py

# Custom sweep:
python aggregate_ptsa_autogen_hdf5.py \
    --mean_rp 0.015 --rel_std_rp 0.10 --k 0.95 \
    --Df_min 2.35 --Df_max 2.95 --num_Df 2 \
    --Np_min 100 --Np_max 400 --num_Np 4 \
    --agg_num 0 1 2

# Use 4 parallel worker processes:
python aggregate_ptsa_autogen_hdf5.py --workers 4

# Tunable cluster-cluster aggregation instead of PTSA:
python aggregate_ptsa_autogen_hdf5.py --method cca \
    --Df_min 1.80 --Df_max 2.50 --num_Df 8 --agg_num 0 1 2

# Show all options:
python aggregate_ptsa_autogen_hdf5.py -h
```

At startup the script displays the parameter grid size, the estimated volume-equivalent radius range (monodisperse approximation: `Rve = mean_rp * Np^(1/3)`), and the number of worker processes.

#### Parallelism (`--workers`)

The `agg_num` loop is parallelised across CPU cores using `ProcessPoolExecutor`. Each worker process handles all `(Df, Np)` combinations for one `agg_num` and writes to a temporary HDF5 file; the main process merges the results in completion order.

| Option | Default | Behaviour |
| --- | --- | --- |
| `--workers N` | all CPU cores | Use `N` worker processes |
| *(omitted)* | `os.cpu_count()` | Automatically capped at `len(agg_num)` |

> **Note**: Because each worker compiles Numba JIT independently, compilation overhead is incurred once per worker process at startup.

#### Choosing the algorithm (`--method`)

| `--method` | Algorithm | Growth |
| --- | --- | --- |
| `ptsa` (default) | Polydisperse tunable sequential aggregation (Singh & Tsotsas, 2022) | one monomer at a time (particle-cluster) |
| `cca` | Tunable cluster-cluster aggregation (Filippov et al., 2000) | small sequential clusters merged pairwise |

**How `cca` works.** Initial clusters of at most `n_SA` monomers are built by tunable sequential aggregation. Clusters are then merged pairwise. The two centroids are placed at the distance that makes the merged cluster obey the fractal law exactly:

```text
N = k * (Rg_F / a)^Df,     Rg_F^2 = a^2 + <|r - r_c|^2>
```

For each merge, one cluster is moved to a random point on that sphere and the other is rotated at random until no monomers overlap and at least one pair is in contact. The search is a translation of `aggregate_gen_CCA.cpp`.

**Extensions beyond the C++ code** (`rel_std_rp = 0` reproduces the C++ formulas exactly):

- *Any Np.* The C++ code allows only `Np = n_SA * 2^levels`. Here a cluster of N monomers is built from clusters of `floor(N/2)` and `ceil(N/2)` monomers, recursively. For `Np = n_SA * 2^L` this is the C++ hierarchy.
- *Polydisperse monomers.* Radii follow the same distribution as in `ptsa`. Centroids and `Rg_F` are mass weighted, with the self term `r_i^2`. The length scale `a` of a cluster is its mass-weighted rms monomer radius, `a^2 = sum(m_i r_i^2) / sum(m_i)`. A fixed `a = mean_rp` makes small clusters geometrically impossible when their monomers are larger than the mean.
- *Failure handling.* The C++ code loops forever or aborts. Here two cases are distinguished:
  - A step that is geometrically impossible is detected before the random search. This is typical at low Df, where the required centroid distance exceeds the reach of the two clusters. The sub-cluster is then rebuilt from scratch with new radii (at most `max_rebuild = 100` times). Redrawing the whole sub-cluster is a rejection step, so the aggregates follow the same distribution as restarting the whole aggregate.
  - An exhausted random search is typical at high Df. The aggregate returns `Np_final < Np` and is retried up to `--max_try` times, as with `ptsa`.
- *Speed.* A cell list accelerates the contact test. It gives bit-identical results to the brute-force test. On one core, Np = 2048 takes about 1–3 s and Np = 21,720 takes 10–30 s (Df = 1.8–2.35) or 1–2 min (Df = 2.55).

**CCA options** (ignored for `ptsa`):

| Option | Default | Meaning |
| --- | --- | --- |
| `--cca_n_SA` | 4 | max monomers in the initial clusters (C++ `num_sph_SA`) |
| `--cca_tol` | 1e-2 | contact window: surface gap < `cca_tol * mean_rp` (C++ `tol` with a = 1) |
| `--cca_iter1max` | 1000 | max random translations per merge |
| `--cca_iter2max` | 10000 | max random rotations per translation |

`--max_search` limits the trials per monomer in the initial clusters.

**Comparison at the same (k, Df, Np)** (k = 0.95, rel_std_rp = 0.10, Np = 2048; mean ± standard error of 30 aggregates, same seeds for both methods):

| Df | Method | Rg / mean_rp | eps_agg | k_eff (a = a_c) | Contacts per monomer | Gyration-tensor eigenvalue ratio (max/min) |
| --- | --- | --- | --- | --- | --- | --- |
| 2.35 | ptsa | 27.01 ± 0.02 | 0.9504 | 0.952 ± 0.001 | 2.030 ± 0.001 | 4.09 ± 0.26 |
| 2.35 | cca | 27.03 ± 0.01 | 0.9505 | 0.950 (exact) | 2.015 ± 0.001 | 4.42 ± 0.22 |
| 2.55 | ptsa | 20.91 ± 0.01 | 0.8929 | 0.951 ± 0.001 | 2.033 ± 0.001 | 2.00 ± 0.08 |
| 2.55 | cca | 20.92 ± 0.01 | 0.8931 | 0.950 (exact) | 2.030 ± 0.001 | 3.15 ± 0.08 |

`k_eff = Np / (Rg_F / a_c)^Df` is computed with `tcca.fractal_prefactor_filippov`. Contacts are pairs with a surface gap below 0.02 mean_rp. In this test, `Rg` and `eps_agg` of the two methods agree to within 0.1 %. This agreement is empirical: `ptsa` uses a count-based law with the radius of the newly added monomer, not the `a_c` law of `cca`. At Df = 2.55 the CCA aggregates are clearly more elongated. At Df = 2.35 the difference in elongation is within the scatter, and CCA has slightly fewer contacts. Relative to `a = mean_rp` instead of `a_c`, k_eff is about 0.88 for both methods.

**Speed** (k = 0.95, rel_std_rp = 0.10; one aggregate per process, wall time of one successful try):

| Np | Df | ptsa | cca |
| --- | --- | --- | --- |
| 2048 | 2.35 | 13.0 s | 3.2 s |
| 2048 | 2.55 | 6.0 s | 5.8 s |
| 21,720 | 1.8 / 2.0 / 2.35 | not measured | 19 / 13 / 12 s (median of 3) |
| 21,720 | 2.55 | not measured | 79 s (median of 3, max 140 s) |

The Np = 2048 rows are means of 30 aggregates with the same seeds for both methods. All times were measured on an i9-14900K while about 30 of its 32 threads were busy, so compare ratios rather than absolute values. The CCA time grows with Df because the two clusters must interpenetrate more deeply. With `--method cca` at Df ≥ 2.7, one failed try takes 3–50 s and is repeated `--max_try` times.

**Feasible range of `cca`** (k = 0.95, rel_std_rp = 0.10; successes out of 10 seeds, one try each):

| Df | 1.8–2.5 | 2.6 | 2.7 | 2.8 | 2.9 |
| --- | --- | --- | --- | --- | --- |
| Np = 512 | 10/10 | 10/10 | 3/10 | 0/10 | 0/10 |
| Np = 2048 | 10/10 | 9/10 | 0/10 | 0/10 | 0/10 |

Np = 21,720 succeeded for Df = 1.8, 2.0, 2.35 and 2.55 (3/3 each). Above Df ≈ 2.6 the random search finds no non-overlapping configuration at the required centroid distance. A failed try takes 3–50 s. Use `ptsa` for the compact range. The batch script prints a warning when `k + Df_max > 3.5` with `--method cca`.

#### Output files

Output files are written to `./generated_agg_files/`:

| File | Naming rule | Contents |
| --- | --- | --- |
| HDF5 data | `aggregates_YYYYMMDD_xx.h5` (ptsa) / `aggregates_cca_YYYYMMDD_xx.h5` (cca) | monomer coordinates + radii (gzip compressed) |
| Catalog CSV | `catalog_YYYYMMDD_xx.csv` (ptsa) / `catalog_cca_YYYYMMDD_xx.csv` (cca) | one row per aggregate, all parameters + computed properties |

`YYYYMMDD` is the execution date; `xx` is a zero-padded sequential index (00, 01, 02, …) that increments for each run on the same day. The HDF5 file and catalog CSV with the same `xx` always correspond to the same run. PTSA and CCA runs are numbered separately.

#### HDF5 internal structure

Each aggregate is stored as an HDF5 group. The group path follows the parameter ordering convention (constants first, sweep variables after):

```text
aggregates_YYYYMMDD_xx.h5
└── {mean_rp:.4f}/{rel_std_rp:.2f}/{k:.3f}/{Df:.2f}/{Np:05d}/{agg_num}/
    ├── xp   — float64 (Np, 3)  monomer centre positions [μm]
    └── rp   — float64 (Np,)    monomer radii [μm]
    [attrs: mean_rp, rel_std_rp, k, Df, Np, agg_num, Rve, Rg, eps_agg, seed_entropy, method]
    [cca only: cca_n_SA, cca_tol]
```

The catalog CSV columns are: `mean_rp`, `rel_std_rp`, `k`, `Df`, `Np`, `agg_num`, `Rve`, `Rg`, `eps_agg`, `seed_entropy`, `h5_file`, `h5_key`, `method`, (cca only: `cca_n_SA`, `cca_tol`).

`Rg` and `eps_agg` use the same definition for both methods (mass weighted, solid-sphere self term `(3/5) rp^2`). The group key has the same format for both methods, so a PTSA and a CCA aggregate with the same parameters have the same `h5_key`. The method is stored in the file name, in the `method` attribute and in the `method` column. Files written before the CCA option was added have no `method` attribute and are PTSA. The new catalog columns are appended after `h5_key`, so the columns listed in `pcas_lut_schema` keep their order.

`seed_entropy` reproduces an aggregate: `np.random.default_rng(np.random.SeedSequence(int(seed_entropy)))` passed to the generator with the stored parameters gives the identical aggregate.

#### Incremental / interrupted runs

The HDF5 file is opened in append mode (`'a'`). If a group key already exists it is skipped, so interrupted runs can be safely resumed by re-running the script.

---

### Export to MSTM input format (.ptsa)

Convert a stored aggregate to the `.ptsa` CSV format required by the MSTM light-scattering code.

**All parameters are required** (no defaults) to prevent silent mismatches:

```sh
# List available HDF5 files and catalogs:
python export_ptsa_from_hdf5.py --list

# Export a specific aggregate (all 7 arguments are required):
python export_ptsa_from_hdf5.py \
    --h5_file aggregates_20260316_00.h5 \
    --mean_rp 0.020 --rel_std_rp 0.15 --k 0.90 \
    --Df 2.40 --Np 100 --agg_num 0

# Show all options:
python export_ptsa_from_hdf5.py -h
```

If any argument is missing, the error message shows which arguments are needed and the correct parameter order. If the specified key does not exist in the HDF5 file, example keys from the file are displayed to help identify the mismatch.

The exported `.ptsa` file is written to `./generated_agg_files/` with the filename:

```text
meanRp={mean_rp:.3f}um_rstdRp={rel_std_rp:.2f}_k={k:.3f}_Df={Df:.2f}_Np={Np:05d}_agg_num={n}.ptsa
```

Aggregates generated with `--method cca` get the suffix `_method=cca` before `.ptsa`, so they never overwrite a PTSA export with the same parameters. The method is read from the HDF5 attribute.

The filename contains input parameters only. Computed properties (`Rve`, `Rg`, `eps_agg`) are stored in the HDF5 group attributes and `catalog_YYYYMMDD_xx.csv`.

---

### .ptsa file format

Each `.ptsa` file contains `Np` lines of 4 space-delimited floating-point numbers:

```text
x [μm]   y [μm]   z [μm]   rp [μm]
```

Positions are measured from the centroid of the aggregate.

---

### File overview

| File | Purpose |
| --- | --- |
| `aggregate_ptsa_single.ipynb` | Generate and visualise a single aggregate; write `.ptsa` and `.jpg` files |
| `aggregate_ptsa_autogen_hdf5.py` | Batch execution (CLI script) with HDF5 output and catalog CSV; `--method ptsa` or `cca` |
| `tcca.py` | Tunable cluster-cluster aggregation (`tcca`) and geometry diagnostics |
| `export_ptsa_from_hdf5.py` | Export one aggregate from HDF5 to `.ptsa` format (CLI script) |
| `tests/validate_tcca.py` | Validation of `tcca` (geometry, fractal law, C++ formulas, reproducibility): `python tests/validate_tcca.py` |

---

## 📝 License

This project is licensed under the MIT License. See the LICENSE file for details.

## 📖 References

- PTSA algorithm
  - Singh, A. K., & Tsotsas, E. (2022). Influence of polydispersity and breakage on stochastic simulations of spray fluidized bed agglomeration. Chemical Engineering Science, 247, 117022.
- Tunable CCA algorithm
  - Filippov, A. V., Zurita, M., & Rosner, D. E. (2000). Fractal-like aggregates: Relation between morphology and physical properties. Journal of Colloid and Interface Science, 229, 261–273.
  - Moteki, N. "aggregate_gen" — An efficient C++ code for generating fractal cluster of spheres, nmoteki/aggregate_generator (v1.1), GitHub repository, doi:10.5281/zenodo.3567236, 2019.

## 📢 Author

Name: Nobuhiro Moteki
GitHub: @NobuhiroMoteki
Email: <nobuhiro.moteki@gmail.com>
