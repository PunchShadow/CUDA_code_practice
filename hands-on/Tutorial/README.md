## Introduction

I follow the course hosted by Wen-mei Hwu, professor in UIUC

## Architecture

    .
    ├── matrixMul               # Simple matrix multiplcation on GPUs
    ├── vecAdd                  # Adding two vectors with GPUs
    ├── tiledMatrixMul          # Tiled version of matrix multiplication on GPUs
    └── README.md


## Experiments

### Setup
* `hardware`:
    * `CPU`: Intel i9-10900K @ 3.7GHz
    * `RAM`: 64GB DDR4-3600
    * `GPU`: Nvidia RTX 3090


### Matrix Multiplication

* Tiled vs non-Tiled matrix multiplication
```
Matrix A: 1024 x 1024
Matrix B: 1024 x 1024
Grid size: 64 x 64
Block size: 16 x 16

non-tiled version: 1027 us
tiled version: 795 us
```
