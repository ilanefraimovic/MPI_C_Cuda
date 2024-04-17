#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "graph_create.h"
#include <string.h>
#include <stdio.h>

#include "helper_cuda.h"
#ifndef CUDA_BFS_H_
#define CUDA_BFS_H_
extern "C" int execute_bfs(CSR_Graph* graph);
#endif