#include "BFS_Cuda.cuh"
#ifndef MPI_CUDA_BFS_H
#define MPI_CUDA_BFS_H
extern "C" int mpi_cuda_execute_bfs(CSR_Graph* graph, int final_node, int pid);
extern "C" CSR_Graph** partition_graph(CSR_Graph* graph, int num_partitions);
#endif
