# Adaptive RKC (Runge-Kutta-Chebyshev) Method for Stiff ODEs

## Introduction
This project implements an adaptive Runge-Kutta-Chebyshev (RKC) method for solving stiff ordinary differential equations (ODEs) arising from the spatial discretization of partial differential equations (PDEs). 

## Features
- Adaptive time stepping using local error estimation.
- Three error estimation strategies (switch `j1 = 1, 2, 3`).
- Automatic spectral radius estimation for nonlinear problems.


## Dependencies
- Python 3.6 or higher
- NumPy (≥1.19)

Optional: Matplotlib for result visualization.

## Installation
In an Anaconda3 environment, install NumPy via:
```bash
conda install numpy
