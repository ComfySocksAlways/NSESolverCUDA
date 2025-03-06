# Navier-Stokes Solver

This Navier-Stokes solver was developed as part of the course "High End Simulation Practice" at FAU. The solver uses a SIMPLE-like approach, which is a predictor-corrector method. It solves for the velocity by avoiding the pressure initially and then uses a pressure Poisson equation to correct the velocity at the end. Additionally, it employs a staggered grid for better stability.

The implementation leverages CUDA kernels for efficient computation. Custom kernels were written for the stencil update and the pressure Poisson solver, which uses a Conjugate Gradient (CG) solver. Additionally, custom kernels were developed for the CG solver, and the CG solver also utilizes the cuSPARSE library from NVIDIA for matrix-vector products.

## Note

This project is a copy of the original repository located on the university's GitHub page. You can use the provided Makefile to build the project. Building the project requires the Nvidia compiler (nvcc).