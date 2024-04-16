#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "BFS_serial_c.h"

#include <stdio.h>

//vertex centric push BFS from textbook
__global__ void bfs_kernel (unsigned int* graph_row_ptrs,
							unsigned int* graph_dst,
							unsigned int* level,
							unsigned int* newVertexVisited,
							unsigned int* found,
							unsigned int* currLevel,
							unsigned int* graph_num_vertices)
{
  unsigned int vertex = blockIdx.x * blockDim.x + threadIdx.x;
  if (vertex < *graph_num_vertices) {
	if (level[vertex] == *currLevel) {
	  if (vertex == *graph_num_vertices - 1) *found = 1;
	  for (unsigned int edge = graph_row_ptrs[vertex];
		   edge < graph_row_ptrs[vertex + 1];
		   ++edge) {
		unsigned int neighbor = graph_dst[edge];
		if (level[neighbor] == UINT_MAX) {
		  level[neighbor] = *currLevel + 1;
		  *newVertexVisited = 1;
		}
	  }
	}
  }
}

int execute_bfs(CSR_Graph* graph)
{


  	cudaDeviceProp prop;

	int devcount;
	
	// Returns the number of CUDA devices attached to system
	cudaGetDeviceCount(&devcount);

	// Iterate and fetch the details of each deviceID
	for (int i = 0; i < devcount; i++)
	{
		cudaGetDeviceProperties(&prop, i);
		printf("\n\n Name: %s", prop.name);
		printf("\n Multiprocessor count: %d", prop.multiProcessorCount);
		printf("\n Clock rate: %d", prop.clockRate);
		printf("\n Compute Cap: %d.%d", prop.major, prop.minor);
	}

    // Allocate memory for BFS kernel arguments on the device
	unsigned int* device_graph_row_ptrs;
	unsigned int* device_graph_dst;	
	unsigned int* device_graph_num_vertices;
    unsigned int* device_level;
    unsigned int* device_newVertexVisited;
	unsigned int* device_currLevel;
	unsigned int* device_found;
	cudaMalloc((void**)&device_graph_row_ptrs, MAX * sizeof(unsigned int));
	cudaMalloc((void**)&device_graph_dst, MAX * sizeof(unsigned int));
	cudaMalloc((void**)&device_graph_num_vertices, sizeof(unsigned int));
    cudaMalloc((void**)&device_level, graph->num_vertices * sizeof(unsigned int));
    cudaMalloc((void**)&device_newVertexVisited, sizeof(unsigned int));
	cudaMalloc((void**)&device_currLevel, sizeof(unsigned int));
	cudaMalloc((void**)&device_found, sizeof(unsigned int));

    // Initialize level array, newVertexVisited flag, and current level on the host
    unsigned int host_level[graph->num_vertices];
	memset(host_level, UINT_MAX, sizeof(host_level));
    unsigned int host_newVertexVisited = 1;
	unsigned int host_currLevel = 0;
	unsigned int host_found = 0;

    // Copy graph data, level array, newVertexVisited flag from host to device
	cudaMemcpy(device_graph_num_vertices, &graph->num_vertices, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_graph_row_ptrs, graph->rowPtrs, MAX * sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(device_graph_dst, graph->dst, MAX * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_level, host_level, graph->num_vertices * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_newVertexVisited, &host_newVertexVisited, sizeof(unsigned int), cudaMemcpyHostToDevice);
	
    // Define kernel blocksize and num blocks
    int blockSize = 256;
    int numBlocks = (graph->num_vertices + blockSize - 1) / blockSize;

    

	while (host_newVertexVisited && !host_found) {
	  //copy current level and found check to device
	  cudaMemcpy(device_currLevel, &host_currLevel, sizeof(unsigned int), cudaMemcpyHostToDevice);
	  cudaMemcpy(device_found, &host_found, sizeof(unsigned int), cudaMemcpyHostToDevice);
	  
      // Call the BFS kernel
	  bfs_kernel<<<numBlocks, blockSize>>>(device_graph_row_ptrs, device_graph_dst, device_level, device_newVertexVisited, device_found, device_currLevel, device_graph_num_vertices);

      // Wait for kernel to finish
      cudaDeviceSynchronize();

      // Check if any new vertices were visited in this level
      cudaMemcpy(&host_newVertexVisited, device_newVertexVisited, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	  cudaMemcpy(&host_found, device_found, sizeof(unsigned int), cudaMemcpyDeviceToHost);

      // Increment current level
      host_currLevel++;
    }


    // Copy results from device to host if needed

    // Free device memory
    cudaFree(device_level);
    cudaFree(device_newVertexVisited);
    cudaFree(device_currLevel);
    cudaFree(device_found);

    // Free host memory if needed

    return host_found;
}

