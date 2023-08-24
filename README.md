## Introduction

This repository is the collection of my code practices with CUDA parallel programming on GPUs.
I will target on graph algorithms, for instance, BFS (Breadth-First-Search), SSSP (Single-Source-Shortest-Path), in this repository. 
Besides, I will implement some the above algorithms in the state-of-the-art framework, such as [Gunrock](https://github.com/gunrock/gunrock), if the time is available.


## Architecture of the Repository

    .
    ├── hands-on                # All the practice codes
    |   ├── BFS                 # Different graph algorithms
    |   ├── SSSP
    |   └── ...
    ├── Gunrock                 # SOTA directory
    |   ├── ...                 
    |   └── others              
    ├── ...
    ├── data                    # The benchmark (graph datasets)
    |   ├── CSR                 # CSR format of the graph
    |   └── ...             
    ├── LICENSE
    └── README.md
