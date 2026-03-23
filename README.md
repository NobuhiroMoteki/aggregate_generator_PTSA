# aggregate_generator_PTSA

## 📌 Description

A Python code for the polydisperse tunable sequential aggregation (PTSA) algorithm. Monomer positions are sequentially determined by a particle-cluster stochastic aggregation algorithm so as to satisfy the fractal scaling law, surface-attachment condition, and non-overlapping condition. The monomer position is arbitrary, not being restricted to a gridded lattice space. A notable advantage of this PTSA algorithm is its capability of generating compact fractal-like aggregates with high Df (up to 2.95), where the conventional cluster-cluster aggregation (CCA) algorithm will easily break down.

The inner search loop is accelerated by JIT compilation via [Numba](https://numba.pydata.org/) and an early-exit scalar overlap check, achieving a **2.8–3.5× speedup** over a naive vectorized implementation.

### Main Features

- **`ptsa`**: aggregate generator with JIT compilation (Numba) and early-exit scalar overlap check
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

---

## 🚀 Installation

The author developed and tested current aggregate_generator_PTSA (v0.4.0) using Python 3.13.12 in Windows 11 and WSL2 (Ubuntu on Windows 11) machines.

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

#### Output files

Output files are written to `./generated_agg_files/`:

| File | Naming rule | Contents |
| --- | --- | --- |
| HDF5 data | `aggregates_YYYYMMDD_xx.h5` | monomer coordinates + radii (gzip compressed) |
| Catalog CSV | `catalog_YYYYMMDD_xx.csv` | one row per aggregate, all parameters + computed properties |

`YYYYMMDD` is the execution date; `xx` is a zero-padded sequential index (00, 01, 02, …) that increments for each run on the same day. The HDF5 file and catalog CSV with the same `xx` always correspond to the same run.

#### HDF5 internal structure

Each aggregate is stored as an HDF5 group. The group path follows the parameter ordering convention (constants first, sweep variables after):

```text
aggregates_YYYYMMDD_xx.h5
└── {mean_rp:.4f}/{rel_std_rp:.2f}/{k:.3f}/{Df:.2f}/{Np:05d}/{agg_num}/
    ├── xp   — float64 (Np, 3)  monomer centre positions [μm]
    └── rp   — float64 (Np,)    monomer radii [μm]
    [attrs: mean_rp, rel_std_rp, k, Df, Np, agg_num, Rve, Rg, eps_agg]
```

The catalog CSV columns are: `mean_rp`, `rel_std_rp`, `k`, `Df`, `Np`, `agg_num`, `Rve`, `Rg`, `eps_agg`, `h5_file`, `h5_key`.

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
| `aggregate_ptsa_autogen_hdf5.py` | Batch execution (CLI script) with HDF5 output and catalog CSV |
| `export_ptsa_from_hdf5.py` | Export one aggregate from HDF5 to `.ptsa` format (CLI script) |

---

## 📝 License

This project is licensed under the MIT License. See the LICENSE file for details.

## 📖 References

- PTSA algorithm
  - Singh, A. K., & Tsotsas, E. (2022). Influence of polydispersity and breakage on stochastic simulations of spray fluidized bed agglomeration. Chemical Engineering Science, 247, 117022.

## 📢 Author

Name: Nobuhiro Moteki
GitHub: @NobuhiroMoteki
Email: <nobuhiro.moteki@gmail.com>
