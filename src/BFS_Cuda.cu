#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "BFS_serial_c.h"

#include <stdio.h>

__constant__ unsigned int graph_row_ptrs[MAX];
__constant__ unsinged int graph_dst[MAX];
__constant__ unsinged int graph_num_vertices;

//vertex centric push BFS from textbook
__global__ void bfs_kernel (unsigned int* graph_row_ptrs,
							unsigned int* graph_dst,
							unsigned int* level,
							unsigned int* newVertexVisited,
							unsigned int* found;
							unsigned int currLevel,
							unsigned int graph_num_vertices)
{
  unsigned int vertex = blockIdx.x * blockDim.x + threadIdx.x;
  if (vertex < graph_num_vertices) {
	if (level[vertex] == currLevel) {
	  if (vertex == graph_num_vertices - 1) *found = 1;
	  for (unsigned int edge = graph_row_ptrs[vertex];
		   edge < graph_row_ptrs[vertex + 1];
		   ++edge) {
		unsigned int neighbor = graph_dst[edge];
		if (level[neighbor] == UINT_MAX) {
		  level[neighbor] = currLevel + 1;
		  *newVertexVisited = 1;
		}
	  }
	}
  }
}

int execute_bfs(CSR_Graph* graph)
{
    // Copy graph contents to cuda constants
	cudaMemcpyToSymbol(graph_num_vertices, graph->num_vertices, sizeof(unsigned int));
    cudaMemcpyToSymbol(graph_row_ptrs, graph->rowPtrs, MAX * sizeof(unsigned int));
	cudaMemcpyToSymbol(graph_dst, graph->dst, MAX * sizeof(unsigned int));

    // Allocate memory for BFS kernel arguments on the device
    unsigned int* device_level;
    unsigned int* device_newVertexVisited;
	unsigned int* device_currLevel;
	unsigned int* device_found;
    cudaMalloc((void**)&device_level, csrGraph.num_vertices * sizeof(unsigned int));
    cudaMalloc((void**)&device_newVertexVisited, sizeof(unsigned int));
	cudaMalloc((void**)&device_currLevel, sizeof(unsigned int));
	cudaMalloc((void**)&device_found, sizeof(unsigned int));

    // Initialize level array, newVertexVisited flag, and current level on the host
    unsigned int host_level[graph->num_vertices];
	memset(host_level, UINT_MAX, sizeof(host_level));
    unsigned int host_newVertexVisited = 1;
	unsigned int host_currLevel = 0;
	unsinged int host_found = 0;

    // Copy level array, newVertexVisited flag, and current level from host to device
    cudaMemcpy(device_level, host_level, csrGraph.num_vertices * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_newVertexVisited, &host_newVertexVisited, sizeof(unsigned int), cudaMemcpyHostToDevice);
	

    // Define kernel blocksize and num blocks
    int blockSize = 256;
    int numBlocks = (csrGraph.num_vertices + blockSize - 1) / blockSize;

    

	while (host_newVertexVisited && !host_found) {
	  //copy current level to device
	  cudaMemcpy(device_currLevel, &host_currLevel, sizeof(unsigned int), cudaMemcpyHostToDevice);
	  cudaMemcpy(device_found, &host_found, sizeof(unsigned int), cudaMemcpyHostToDevice);
	  
      // Call the BFS kernel
	  bfs_kernel<<<numBlocks, blockSize>>>(graph_row_ptrs, graph_dst, device_level, device_newVertexVisited, device_found, device_currLevel, graph_num_vertices);

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
    cudaFree(device_rowPtrs);
    cudaFree(device_dst);
    cudaFree(device_level);
    cudaFree(device_newVertexVisited);

    // Free host memory if needed

    return host_found;
}






















//vertex centric Frontier-Based BFS from textbook
__global__ void bfs_kernel(CSRGraph csrGraph,
						   unsigned int* level,
						   unsigned int* prevFrontier,
						   unsigned int* currFrontier,
						   unsigned int numPrevFrontier,
						   unsigned int* numCurrFrontier,
						   unsigned int currLevel)
{

    // Initialize privatized frontier
    __shared__ unsigned int currFrontier_s[LOCAL_FRONTIER_CAPACITY];
    __shared__ unsigned int numCurrFrontier_s;
    if(threadIdx.x == 0) {
        numCurrFrontier_s = 0;
    }
    __syncthreads();

    // Perform BFS
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < numPrevFrontier) {
        unsigned int vertex = prevFrontier[i];
        for(unsigned int edge = csrGraph.srcPtrs[vertex]; edge < csrGraph.srcPtrs[vertex + 1]; ++edge) {
            unsigned int neighbor = csrGraph.dst[edge];
            if(atomicCAS(&level[neighbor], UINT_MAX, currLevel) == UINT_MAX) { // Vertex not previously visited
                unsigned int currFrontierIdx_s = atomicAdd(&numCurrFrontier_s, 1);
                if(currFrontierIdx_s < LOCAL_FRONTIER_CAPACITY) {
                    currFrontier_s[currFrontierIdx_s] = neighbor;
                } else {
                    numCurrFrontier_s = LOCAL_FRONTIER_CAPACITY;
                    unsigned int currFrontierIdx = atomicAdd(numCurrFrontier, 1);
                    currFrontier[currFrontierIdx] = neighbor;
                }
            }
        }
    }
    __syncthreads();


    // Allocate in global frontier
    __shared__ unsigned int currFrontierStartIdx;
    if(threadIdx.x == 0) {
        currFrontierStartIdx = atomicAdd(numCurrFrontier, numCurrFrontier_s);
    }
    __syncthreads();

    // Commit to global frontier
    for(unsigned int currFrontierIdx_s = threadIdx.x; currFrontierIdx_s < numCurrFrontier_s;
                                                                                   currFrontierIdx_s += blockDim.x) {
        unsigned int currFrontierIdx = currFrontierStartIdx + currFrontierIdx_s;
        currFrontier[currFrontierIdx] = currFrontier_s[currFrontierIdx_s];
    }
}

int main()
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
}
