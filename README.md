# earth-gravity-disturbing-potential-bem
Boundary Element Method (BEM) solver for Earth’s disturbing gravitational potential using linear triangular elements, with optional OpenMP acceleration and EOC validation.

# Earth Disturbing Potential via Boundary Element Method (BEM)

This project computes the Earth's **disturbing gravitational potential** using a **Boundary Element Method (BEM)** with **linear triangular elements** and **collocation points at mesh vertices**. The resulting linear system has the form:

**F u = G q**

where **u** is the unknown potential, and **q** represents boundary data (e.g., gravity-related loading/flux terms), assembled from BEM integral operators.

## Key features
- Linear triangular surface discretization (vertex collocation).
- Variable vertex neighborhood size due to mesh topology:
  - polar/top-bottom vertices have **4** neighbors,
  - most interior vertices have **6** neighbors.
- Dense matrix assembly of BEM operators (F, G).
- Iterative solver: **BiCGSTAB** for the linear system.
- **Experimental Order of Convergence (EOC)** validation using two mesh resolutions:
  - **902** elements (coarse)
  - **3602** elements (finer)
  - expected EOC ≈ **2** for the chosen discretization.
- Optional **OpenMP** parallelization (used for large runs on an HPC cluster).

## EOC validation (how it works)
To verify correctness, the code can compute an EOC estimate by comparing RMSE errors from the two mesh resolutions (902 vs 3602). After validation, the solver was parallelized to support much larger meshes (e.g., ~160k elements), which would otherwise be too expensive due to dense matrices.

## Build (serial, default)
```bash
clang++ -std=c++17 Source.cpp -O2 -lm -o bem_solver
./bem_solver
