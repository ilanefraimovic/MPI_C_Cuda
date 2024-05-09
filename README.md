# Massively Parallel Breadth-First Search Pathfinding Algorithm
## Overview

This project explores the development and performance of various implementations of the Breadth-First Search (BFS) pathfinding algorithm. By leveraging CPU and GPU parallel computing, we compare four distinct implementations to assess their speed and efficiency across different grid sizes.
Implementations
```
    Serial C Implementation: A baseline implementation in C++ to provide a reference point for comparison.
    MPI Implementation: Distributed grid traversal using the MPI framework.
    Single CUDA Implementation: Parallel BFS on a single GPU using CUDA.
    CUDA + MPI Implementation: A hybrid approach combining MPI for inter-GPU communication and CUDA for intra-GPU parallel processing.
```
## Objectives

    Increase the speed and efficiency of the BFS algorithm.
    Optimize resource utilization on GPU clusters.
    Perform detailed performance comparisons of CPU, MPI, GPU, and hybrid implementations.

## Project Structure

    src/: Contains source code for each BFS implementation.
        - Source files for the Serial C/ MPI C implementation.
        - Source files for the Single CUDA implementation.
        - Source files for the combined CUDA + MPI implementation.
        - Driver file for all besides MPI implementation
    src2/: Contains source code driver for MPI implementation.
    Screenshots/: Screenshots of Output of Program
        - Long format output- printing each node on a graph of 16 nodes for each type of BFS implementation
        - Short format output- printing runtime for each type of BFS implementation for graphs of size 10000, 250000, 1000000, 4000000
## Setup Instructions
### Prerequisites
```
    C++ Compiler (GCC/Clang recommended)
    MPI Library (OpenMPI or MPICH)
    CUDA Toolkit (version 10.0 or higher)
    Passwordless SSH between nodes
    NFS set up for shared directory
    Create a hostfile that contains the IP adress of other nodes in the Cluster named mpi_hostfile. Move it to src directory and copy to src2.

```
### Build/Run Instructions
#### Serial C / Cuda / Cuda + MPI Implementation:
    Navigate to src directory
    type ```make```
    Either:
    type ```mpirun -np x --hostfile mpi_hostfile ./BFS y'''
    where x is the number of nodes in the cluster and y is the number of rows and columns in the grid-like graph.
    OR:
    type '''./run.sh'''
    to run the program multiple times with graphs of size 10000, 250000, 1000000, 4000000

#### MPI Implementation:

    Navigate to src2 directory
    type ```make```
    Either:
    type ```mpirun -np x --hostfile mpi_hostfile ./BFS y'''
    where x is the number of nodes in the cluster and y is the number of rows and columns in the grid-like graph.
    OR:
    type '''./run.sh'''
    to run the program multiple times with graphs of size 10000, 250000, 1000000, 4000000

## Future Work

    Implement the A* search algorithm for heuristic-based pathfinding.
    Explore other parallel computing frameworks to further enhance BFS efficiency.
    Optimize memory management and communication in CUDA and MPI hybrid approaches.
