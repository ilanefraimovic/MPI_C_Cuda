#include "BFS_Cuda.cuh"

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
	  printf("Cuda Visited: %d\n", vertex);
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

int cuda_execute_bfs(CSR_Graph* graph)
{
  printf("Hello from the beginning of execute_bfs\n");
    // Allocate memory for BFS kernel arguments on the device
	unsigned int* device_graph_row_ptrs;
	unsigned int* device_graph_dst;	
	unsigned int* device_graph_num_vertices;
    unsigned int* device_level;
    unsigned int* device_newVertexVisited;
	unsigned int* device_currLevel;
	unsigned int* device_found;
	checkCudaErrors(cudaMalloc((void**)&device_graph_row_ptrs, graph->num_vertices * sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc((void**)&device_graph_dst, graph->num_edges * sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc((void**)&device_graph_num_vertices, sizeof(unsigned int)));
    checkCudaErrors(cudaMalloc((void**)&device_level, graph->num_vertices * sizeof(unsigned int)));
    checkCudaErrors(cudaMalloc((void**)&device_newVertexVisited, sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc((void**)&device_currLevel, sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc((void**)&device_found, sizeof(unsigned int)));
    // Initialize level array, newVertexVisited flag, and current level on the host
	unsigned int* host_level = (unsigned int*)malloc(graph->num_vertices * sizeof(unsigned int));
	if (host_level == NULL) {
	  printf("Unable to allocate host array in cuda BFS\n");
	  exit(1);
	}
	memset(host_level, UINT_MAX, graph->num_vertices * sizeof(unsigned int));
	host_level[0] = 0;
    unsigned int host_newVertexVisited = 1;
	unsigned int host_currLevel = 0;
	unsigned int host_found = 0;

    // Copy graph data, level array, newVertexVisited flag from host to device
	checkCudaErrors(cudaMemcpy(device_graph_num_vertices, &graph->num_vertices, sizeof(unsigned int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_graph_row_ptrs, graph->rowPtrs, graph->num_vertices * sizeof(unsigned int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(device_graph_dst, graph->dst, graph->num_edges * sizeof(unsigned int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_level, host_level, graph->num_vertices * sizeof(unsigned int), cudaMemcpyHostToDevice));
	
    // Define kernel blocksize and num blocks
    int blockSize = 256;
    int numBlocks = (graph->num_vertices + blockSize - 1) / blockSize;

	printf("Hello from right before kernel call\n");

	while (host_newVertexVisited && !host_found) {
	  //copy current level and found check to device
	  checkCudaErrors(cudaMemcpy(device_currLevel, &host_currLevel, sizeof(unsigned int), cudaMemcpyHostToDevice));
	  checkCudaErrors(cudaMemcpy(device_found, &host_found, sizeof(unsigned int), cudaMemcpyHostToDevice));

	  //reset new vertex visited
	  host_newVertexVisited = 0;
	  checkCudaErrors(cudaMemcpy(device_newVertexVisited, &host_newVertexVisited, sizeof(unsigned int), cudaMemcpyHostToDevice));
	  
      // Call the BFS kernel
	  bfs_kernel <<<numBlocks, blockSize>>>(device_graph_row_ptrs, device_graph_dst, device_level, device_newVertexVisited, device_found, device_currLevel, device_graph_num_vertices);

	  // Check for kernel launch errors
	  cudaError_t cudaError = cudaGetLastError();
	  if (cudaError != cudaSuccess) {
		fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(cudaError));
		// Handle the error appropriately (e.g., clean up and exit)
		exit(EXIT_FAILURE);
	  }

      // Wait for kernel to finish
      cudaDeviceSynchronize();
	  
      // Check if any new vertices were visited in this level
      checkCudaErrors(cudaMemcpy(&host_newVertexVisited, device_newVertexVisited, sizeof(unsigned int), cudaMemcpyDeviceToHost));
	  checkCudaErrors(cudaMemcpy(&host_found, device_found, sizeof(unsigned int), cudaMemcpyDeviceToHost));
	  
	  checkCudaErrors(cudaMemcpy(host_level, device_level, graph->num_vertices * sizeof(unsigned int), cudaMemcpyDeviceToHost));
	  printf("we are here\n");
      // Increment current level
      host_currLevel++;
	  printf("-------------------------------------\n");
    }


    // Copy results from device to host if needed

    // Free device memory
    cudaFree(device_level);
    cudaFree(device_newVertexVisited);
    cudaFree(device_currLevel);
    cudaFree(device_found);

    // Free host memory if needed
	free(host_level);
    return host_found;
}
/*
int main() {
  printf("Hello from the beginning of main\n");
  CSR_Graph* graph = create_grid_like_csr_graph(3);
  printCSR(graph);
  execute_bfs(graph);
  return 0;
}
*/
