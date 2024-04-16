#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "BFS_serial_c.h"

#include <stdio.h>
typedef struct CSRGraph {
  unsigned int num_vertices;
  unsigned int rowPtrs[];
  unsigned int dst[];
}
//vertex centric push BFS from textbook
__global__ void bfs_kernel (CSRGraph csrGraph,
							unsigned int* level,
							unsigned int* newVertexVisited,
							unsigned int currLevel)
{
  unsigned int vertex = blockIdx.x * blockDim.x + threadIdx.x;
  if (vertex < csrGraph.num_vertices) {
	if (level[vertex] == currLevel) {
	  for (unsigned int edge = csrGraph.rowPtrs[vertex];
		   edge < csrGraph.rowPtrs[vertex + 1];
		   ++edge) {
		unsigned int neighbor = csrGraph.dst[edge];
		if (level[neighbor] == UINT_MAX) {
		  level[neighbor] = currLevel + 1;
		  *newVertexVisited = 1;
		}
	  }
	}
  }
}

int execute_bfs()
{
    // Define graph properties
    CSRGraph csrGraph;
    csrGraph.num_vertices = 100; // Example value

    // Allocate and initialize graph data on the host
    unsigned int* host_rowPtrs = /* Your host rowPtrs initialization */;
    unsigned int* host_dst = /* Your host dst initialization */;

    // Allocate memory for graph data on the device
    unsigned int* device_rowPtrs;
    unsigned int* device_dst;
    cudaMalloc((void**)&device_rowPtrs, (csrGraph.num_vertices + 1) * sizeof(unsigned int));
    cudaMalloc((void**)&device_dst, /* Calculate size based on number of edges */);

    // Copy graph data from host to device
    cudaMemcpy(device_rowPtrs, host_rowPtrs, (csrGraph.num_vertices + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_dst, host_dst, /* Size based on number of edges */, cudaMemcpyHostToDevice);

    // Allocate memory for BFS kernel arguments on the device
    unsigned int* device_level;
    unsigned int* device_newVertexVisited;
    cudaMalloc((void**)&device_level, csrGraph.num_vertices * sizeof(unsigned int));
    cudaMalloc((void**)&device_newVertexVisited, sizeof(unsigned int));

    // Initialize level array and newVertexVisited flag on the host
    unsigned int* host_level = /* Your initialization */;
    unsigned int host_newVertexVisited = /* Your initialization */;

    // Copy level array and newVertexVisited flag from host to device
    cudaMemcpy(device_level, host_level, csrGraph.num_vertices * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_newVertexVisited, &host_newVertexVisited, sizeof(unsigned int), cudaMemcpyHostToDevice);

    // Define kernel launch configuration
    int blockSize = 256;
    int numBlocks = (csrGraph.num_vertices + blockSize - 1) / blockSize;

    

	while (anyNewVerticesVisited) {
      // Call the BFS kernel
	  bfs_kernel<<<numBlocks, blockSize>>>(csrGraph, device_level, device_newVertexVisited, /* Pass currLevel */);

      // Wait for kernel to finish
      cudaDeviceSynchronize();

      // Check if any new vertices were visited in this level
      cudaMemcpy(&anyNewVerticesVisited, newVertexVisited, sizeof(bool), cudaMemcpyDeviceToHost);

      // Increment current level
      currLevel++;
    }


    // Copy results from device to host if needed

    // Free device memory
    cudaFree(device_rowPtrs);
    cudaFree(device_dst);
    cudaFree(device_level);
    cudaFree(device_newVertexVisited);

    // Free host memory if needed

    return 0;
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
