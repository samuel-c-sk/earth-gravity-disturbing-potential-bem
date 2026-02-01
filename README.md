# earth-gravity-disturbing-potential-bem

Boundary Element Method (BEM) solver for Earth’s disturbing gravitational potential using linear triangular elements, with optional OpenMP acceleration and EOC validation.

## Earth Disturbing Potential via Boundary Element Method (BEM)

This project computes the Earth's **disturbing gravitational potential** using a **Boundary Element Method (BEM)** with **linear triangular elements** and **collocation points at mesh vertices**. The resulting linear system has the form:

**F u = G q**

where **u** is the unknown potential, and **q** represents boundary data (e.g., gravity-related loading / flux terms), assembled from BEM integral operators.

## Key features
- Linear triangular surface discretization (vertex collocation).
- Variable vertex neighborhood size due to mesh topology:
  - polar/top-bottom vertices have **4** neighbors,
  - most interior vertices have **6** neighbors.
- Dense matrix assembly of BEM operators (F, G).
- Iterative solver: **BiCGSTAB** for the linear system.
- **Experimental Order of Convergence (EOC)** validation using two mesh resolutions:
  - **902** elements (coarse),
  - **3602** elements (finer),
  - expected EOC ≈ **2** for the chosen discretization.
- Optional **OpenMP** parallelization (used for large runs on an HPC cluster).

## EOC validation (how it works)
To verify correctness, the code can compute an EOC estimate by comparing RMSE errors from the two mesh resolutions (902 vs 3602). After validation, the solver was parallelized to support much larger meshes (e.g., ~160k elements), which would otherwise be too expensive due to dense matrices.

## Source code
The main entry point of the solver is located in:
src/Source.cpp

This file contains the full BEM formulation, matrix assembly, BiCGSTAB solver, optional OpenMP parallelization, and EOC validation logic.

## Input data
The solver expects external input files describing:
- mesh geometry and boundary data (vertex coordinates, heights, boundary values),
- element connectivity and vertex neighborhoods.

⚠️ **Note:**  
The original datasets used during coursework (e.g. meshes with 902 and 3602 elements) are **not included** in this repository, as they may not be redistributable.  
The `data/` directory is provided only as a placeholder for compatible input files.

## Build

### Serial (default)
```bash
clang++ -std=c++17 src/Source.cpp -O2 -lm -o bem_solver
./bem_solver
```
### Parallel (OpenMP)
```bash
gcc src/Source.cpp -O2 -fopenmp -DUSE_OPENMP -lm -o bem_solver
./bem_solver
```
## Output
- data1.dat: computed disturbing potential values at mesh vertices,
- optional output files used for EOC validation.

## Background
This project was developed within a university course on the Boundary Element Method,
focusing on gravitational field modeling and large dense linear systems.
