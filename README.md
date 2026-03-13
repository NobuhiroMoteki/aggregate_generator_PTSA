# aggregate_generator_PTSA

## 📌 Description
A Python code for the polydisperse tunable sequential aggregation (PTSA) algorithm. Monomer positions are sequentially determined by a particle-cluster stochastic aggregation algorithm so as to satisfy the fractal scaling law, surface-attachment condition, and non-overlapping condition. The monomer position is arbitrary, not being restricted to a gridded lattice space. A notable advantage of this PTSA algorithm is its capability of generating compact fractal-like aggregates with high Df (up to 2.95), where the conventional cluster-cluster aggregation (CCA) algorithm will easily break down.

The inner search loop is accelerated by JIT compilation via [Numba](https://numba.pydata.org/) and an early-exit scalar overlap check, achieving a **2.8–3.5× speedup** over a naive vectorized implementation.

### Main Features
- **`ptsa`**: aggregate generator with JIT compilation (Numba) and early-exit scalar overlap check
- Aggregate generator function (ptsa) with inputs and outputs defined as follows:

```
def ptsa(Np, mean_rp, rel_std_rp, Df, k, max_search_num, rng)
    Polydisperse tunable sequential aggregation (PTSA) method for generating fractal-like aggregate.
    Monomer radius follows a normal distribution with specified mean and standard deviation.

    ==== Input parameters ===
    Np: number of monomers
    mean_rp: mean of the normal distribution of the monomer radius
    rel_std_rp: relative standard deviation of the normal distrituion of the monomer radius
    Df: fractal dimension
    k: fractal prefactor
    max_search_num: maximum number of iteration for searching a location of each monomer attached onto the surface of an aggregate (= 20000000)
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

The author developed and tested current aggregate_generator_PTSA (v0.3.1) using Python 3.13.12 in Windows 11 and WSL2 (Ubuntu on Windows 11) machines.

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

### Single execution

1. Open `aggregate_ptsa_single.ipynb` and specify the following parameters in the 4th cell:
   - mean monomer radius [μm]: `mean_rp`
   - relative standard deviation of monomer radius: `rel_std_rp`
   - number of monomers: `Np`
   - fractal dimension: `Df`
   - fractal prefactor: `k`
2. Execute all cells (a 3D scatter plot of the generated aggregate will appear).
3. The output `.ptsa` file is written to `./generated_agg_files/`.

---

### Batch execution (parameter sweep) — HDF5 output

Use `aggregate_ptsa_autogen_hdf5.ipynb` to generate a large number of aggregates over a parameter grid and store them efficiently in a single HDF5 file.

#### Step 1 — Set parameters (cell-4)

| Parameter | Description |
| --- | --- |
| `mean_rp` | mean monomer radius [μm] |
| `rel_std_rp` | relative std of monomer radius |
| `Np_min`, `Np_max`, `num_Np` | linspace grid for number of monomers |
| `Df_min`, `Df_max`, `num_Df` | linspace grid for fractal dimension |
| `k` | fractal prefactor |
| `agg_num_arr` | list of random-aggregate indices (e.g. `[0,1,2]` for 3 realisations per parameter set) |

#### Step 2 — Execute cell-6

Output files are written to `./generated_agg_files/`:

| File | Naming rule | Contents |
| --- | --- | --- |
| HDF5 data | `aggregates_YYYYMMDD_xx.h5` | monomer coordinates + radii (gzip compressed) |
| Catalog CSV | `catalog_YYYYMMDD_xx.csv` | one row per aggregate, all parameters + computed properties |

`YYYYMMDD` is the execution date; `xx` is a zero-padded sequential index (00, 01, 02, …) that increments for each run on the same day. The HDF5 file and catalog CSV with the same `xx` always correspond to the same run.

#### HDF5 internal structure

Each aggregate is stored as an HDF5 group identified by its input parameters:

```text
aggregates_YYYYMMDD_xx.h5
└── {agg_num}/{k:.3f}/{Df:.2f}/{mean_rp:.4f}/{rel_std_rp:.2f}/{Np:05d}/
    ├── xp   — float64 (Np, 3)  monomer centre positions [μm]
    └── rp   — float64 (Np,)    monomer radii [μm]
    [attrs: agg_num, k, Df, mean_rp, rel_std_rp, Np, Rve, Rg, eps_agg]
```

The catalog CSV columns are: `agg_num`, `k`, `Df`, `mean_rp`, `rel_std_rp`, `Np`, `Rve`, `Rg`, `eps_agg`, `h5_file`, `h5_key`.

#### Incremental / interrupted runs

The HDF5 file is opened in append mode (`'a'`). If a group key already exists it is skipped, so interrupted runs can be safely resumed by re-executing cell-6.

---

### Export to MSTM input format (.ptsa)

To convert a stored aggregate to the `.ptsa` CSV format required by the MSTM light-scattering code:

1. Open `export_ptsa_from_hdf5.ipynb`.
2. Execute cell-2 once to define the helper functions.
3. Execute cell-3: available HDF5 files and catalog CSVs are listed automatically.
4. Edit `h5_fname` and the aggregate parameters (`agg_num`, `k`, `Df`, `mean_rp`, `rel_std_rp`, `Np`) in cell-3 and re-execute.

The exported `.ptsa` file is written to `./generated_agg_files/` with the filename:

```text
agg_num={n}_k={k:.3f}_Df={Df:.2f}_meanRp={mean_rp:.3f}um_rstdRp={rel_std_rp:.2f}_Np={Np:05d}.ptsa
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

### Notebook overview

| Notebook | Purpose |
| --- | --- |
| `aggregate_ptsa_single.ipynb` | Generate and visualise a single aggregate; write one `.ptsa` file |
| `aggregate_ptsa_autogen_hdf5.ipynb` | Batch execution with HDF5 output and catalog CSV |
| `export_ptsa_from_hdf5.ipynb` | Export one aggregate from HDF5 to `.ptsa` format for MSTM |

---

## 📝 License
This project is licensed under the MIT License. See the LICENSE file for details.

## 📖 References
- PTSA algorithm
    - Singh, A. K., & Tsotsas, E. (2022). Influence of polydispersity and breakage on stochastic simulations of spray fluidized bed agglomeration. Chemical Engineering Science, 247, 117022.



## 📢 Author
Name: Nobuhiro Moteki
GitHub: @NobuhiroMoteki
Email: nobuhiro.moteki@gmail.com


