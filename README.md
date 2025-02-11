# aggregate_generator_PTSA

## 📌 Description
A python code for the polydisperse tunable sequential aggregation (PTSA) algorithm. Momomer position is sequencially determined by a particle-cluster stochastic aggregation algorithm so as to satisfy all of the fractal scaling raw, surface-attachment condition, and non-overlapping condition. The monomer position is arbitrary, not being restricted to a gridded lattice space. A notable advantage of this PTSA algorithm is its capability of generating compact fractal-like aggregates with high Df (up to 2.95), where the conventional cluster-cluster aggregation (CCA) algorithm will easily break down.

### Main Features
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

The aggregate generation might fail depending on the combination of (k, Df). N.Moteki tested only k=0.95, and Df= 2.35~2.95, 


---

## 🚀 Installation

The author developed and tested current aggregate_generator_PTSA (v0.1.1) using Python 3.12.8 in Windows 11 machines.

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

1. Open the JupyterNotebook file `aggregate_ptsa_jit_single.ipynb` and specify the user input value of the following  parameters in the 4th cell:
   - mean monomer radius [μm]: `mean_rp` 
   - relative standard deviation of monomer radius: `rel_std_rp`
   - number of monomers: `Np`
   - fractal dimension: `Df`
   - fractal prefactor: `k`
2. Execute the JupyterNotebook `aggregate_ptsa_jit_single.ipynb` (A 3D scatter plot of the generated aggregate will appear).

### Many executions (parameter sweep)
1. Open the JupyterNotebook file `aggregate_ptsa_jit_batch.ipynb` and specify the user input value of the following  parameters in the 4th cell:
   - mean monomer radius [μm]: `mean_rp` 
   - relative standard deviation of monomer radius: `rel_std_rp`
   - sweep condition for the number of monomers (linspace grid): `Np_min`, `Np_max`, `num_Np`
   - sweep condition for fractal dimension (logspace grid): `Df_min`, `Df_max`, `num_Df`
   - fractal prefactor: `k`
   - index list of random aggregates : `agg_num_arr`
  The index is used to discriminate aggregates with same set of parameters but with different random seeds. For example, set `agg_num_arr = [0,1,2]` if you generate 3 random aggregates for each set of parameters.
2. Execute the JupyterNotebook `aggregate_ptsa_jit_batch.ipynb`. The generated aggregate files will be stored in the subfolder `.\generated_agg_files`.

### Output file format
Filename indicates the set of parameter values of the aggregate.
In single-execution case, we may have output file named
`agg_k=0.900_Df=2.90_meanRp=0.020um_rstdRp=0.10_Np=00200_Rve=0.119um_Rg=0.133um_epsagg=0.668.ptsa`,
where, the `Rve`, `Rg`, and `epsagg` respectively denote volume equivalent radius, gyration radius, and porosity of the generated aggregate.

In batch-execution case, we may have output file named
`agg_num=4_k=0.950_Df=2.55_meanRp=0.020um_rstdRp=0.10_Np=00020_Rve=0.055um_Rg=0.065um_epsagg=0.716.ptsa`,
where, the `agg_num` denotes the index of random aggregate.

Each `.ptsa` file contains `Np` lines of 4 tab-delimited floating point numbers respectively indicating the monomer's center position and radius: `x  y  z  rp`. The position is measured from the centrod of the aggregate. The length unit is [μm].

---

## 📝 License
This project is licensed under the MIT License. See the LICENSE file for details.

## 📖 References
- PTSA
    - Singh, A. K., & Tsotsas, E. (2022). Influence of polydispersity and breakage on stochastic simulations of spray fluidized bed agglomeration. Chemical Engineering Science, 247, 117022.



## 📢 Author
Name: Nobuhiro Moteki
GitHub: @NobuhiroMoteki
Email: nobuhiro.moteki@gmail.com


