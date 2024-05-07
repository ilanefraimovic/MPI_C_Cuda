#include "graph_create.h"
#include "BFS_MPI_Cuda.cuh"

//vertex centric push BFS from textbook
__global__ void bfs_kernel (unsigned int* graph_row_ptrs,
                            unsigned int* graph_dst,
                            unsigned int* level,
                            unsigned int* newVertexVisited,
                            unsigned int* found,
                            unsigned int* currLevel,
                            unsigned int* graph_num_vertices,
                            unsigned int* final_node,
                            unsigned int* offset)
{
  unsigned int vertex = blockIdx.x * blockDim.x + threadIdx.x;
  if (vertex < *graph_num_vertices) {
    if (level[vertex] == *currLevel) {
      printf("MPI + Cuda Visited: %d\n", vertex + *offset);
      if (vertex + *offset == *final_node) *found = 1;
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
int mpi_cuda_execute_bfs(CSR_Graph* graph, int final_node, int pid) {
    // Allocate memory for BFS kernel arguments on the device
    unsigned int* device_graph_row_ptrs;
    unsigned int* device_graph_dst; 
    unsigned int* device_graph_num_vertices;
    unsigned int* device_level;
    unsigned int* device_newVertexVisited;
    unsigned int* device_currLevel;
    unsigned int* device_found;
    unsigned int* device_final_node;
    unsigned int* device_offset;
    checkCudaErrors(cudaMalloc((void**)&device_graph_row_ptrs, graph->num_vertices * sizeof(unsigned int)));
    checkCudaErrors(cudaMalloc((void**)&device_graph_dst, graph->num_edges * sizeof(unsigned int)));
    checkCudaErrors(cudaMalloc((void**)&device_graph_num_vertices, sizeof(unsigned int)));
    checkCudaErrors(cudaMalloc((void**)&device_level, graph->num_vertices * sizeof(unsigned int)));
    checkCudaErrors(cudaMalloc((void**)&device_newVertexVisited, sizeof(unsigned int)));
    checkCudaErrors(cudaMalloc((void**)&device_currLevel, sizeof(unsigned int)));
    checkCudaErrors(cudaMalloc((void**)&device_found, sizeof(unsigned int)));
    checkCudaErrors(cudaMalloc((void**)&device_final_node, sizeof(unsigned int)));
    checkCudaErrors(cudaMalloc((void**)&device_offset, sizeof(unsigned int)));
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
    checkCudaErrors(cudaMemcpy(device_final_node, &final_node, sizeof(unsigned int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_offset, &graph->offset, sizeof(unsigned int), cudaMemcpyHostToDevice));
    // Define kernel blocksize and num blocks
    int blockSize = 256;
    int numBlocks = (graph->num_vertices + blockSize - 1) / blockSize;


    while (host_newVertexVisited && !host_found) {

      //copy current level and found check to device
      checkCudaErrors(cudaMemcpy(device_currLevel, &host_currLevel, sizeof(unsigned int), cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy(device_found, &host_found, sizeof(unsigned int), cudaMemcpyHostToDevice));

      //reset new vertex visited
      host_newVertexVisited = 0;
      checkCudaErrors(cudaMemcpy(device_newVertexVisited, &host_newVertexVisited, sizeof(unsigned int), cudaMemcpyHostToDevice));
      
      // Call the BFS kernel
      bfs_kernel <<<numBlocks, blockSize>>>(device_graph_row_ptrs, device_graph_dst, device_level, device_newVertexVisited, device_found, device_currLevel, device_graph_num_vertices, device_final_node, device_offset);

      printf("we are here\n----------------PID: %d-----------\n", pid);
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
      
      checkCudaErrors(cudaMemcpy(&host_found, device_found, sizeof(unsigned int), cudaMemcpyDeviceToHost));
      checkCudaErrors(cudaMemcpy(&host_newVertexVisited, device_newVertexVisited, sizeof(unsigned int), cudaMemcpyDeviceToHost));
      
      checkCudaErrors(cudaMemcpy(host_level, device_level, graph->num_vertices * sizeof(unsigned int), cudaMemcpyDeviceToHost));
      // Increment current level
      host_currLevel++;
    }

    // Free device memory
    cudaFree(device_level);
    cudaFree(device_newVertexVisited);
    cudaFree(device_currLevel);
    cudaFree(device_found);
    cudaFree(device_final_node);
	cudaFree(device_graph_num_vertices);
	cudaFree(device_graph_dst);
	cudaFree(device_offset);

    // Free host memory
	free(host_level);
    return host_found;

}

CSR_Graph** partition_graph(CSR_Graph* graph, int num_partitions) {
  //check if the number of partitions is valid
    if (num_partitions <= 1) {
        printf("Number of partitions should be greater than 1.\n");
        exit(1);
    }

    //allocate memory for array of partitioned graphs
    CSR_Graph** subgraphs = (CSR_Graph**)malloc(num_partitions * sizeof(CSR_Graph));
    if (subgraphs == NULL) {
        printf("Memory allocation failed.\n");
        exit(1);
    }

    //calculate the number of vertices per partition
	unsigned int vertices_per_partition = (graph->num_vertices / num_partitions);
	while (vertices_per_partition % (int)sqrt((double)graph->num_vertices) != 0) vertices_per_partition++; 
	//initialize each partitioned graph
    for (int i = 0; i < num_partitions; i++) {
	    subgraphs[i] = (CSR_Graph*)malloc(sizeof(CSR_Graph)); // allocate memory for CSR_Graph
		if (subgraphs[i] == NULL) {
		  printf("Memory allocation failed.\n");
		  exit(1);
		}
        subgraphs[i]->num_vertices = 0;
        subgraphs[i]->rowPtrs = (unsigned int*)malloc((vertices_per_partition + 1) * sizeof(unsigned int));
        subgraphs[i]->num_edges = 0;
        subgraphs[i]->dst = (unsigned int*)malloc((vertices_per_partition * 2) * sizeof(unsigned int));
        subgraphs[i]->last_row_ptr = 0;
		subgraphs[i]->offset = 0;
    }
	

	//partitioning logic
	int current_offset_src = 0;
	int current_offset_dst = 0;
	int last_offset_dst = 0;
	for (unsigned int i = 0, j = 0, k = 0; i < graph->num_vertices; i++, j++, k++) {
	  int current_partition = i / vertices_per_partition;
	  if (j == vertices_per_partition) {
		last_offset_dst = current_offset_dst;
		current_offset_dst += vertices_per_partition;
		while (k < graph->rowPtrs[i]) {
		  subgraphs[current_partition - 1]->dst[j++] = graph->dst[k++] - last_offset_dst;
		  subgraphs[current_partition - 1]->num_edges++;
		}
		current_offset_src = graph->rowPtrs[i];
		subgraphs[current_partition]->offset = current_offset_dst;
		j = 0;
	  }
	  
	  subgraphs[current_partition]->rowPtrs[j] = graph->rowPtrs[i] - current_offset_src;
	  subgraphs[current_partition]->num_vertices++;
	  subgraphs[current_partition]->dst[j] = graph->dst[k] - current_offset_dst;
	  subgraphs[current_partition]->num_edges++;
	  if (i == graph->num_vertices - 1) {
		while (k < graph->num_edges) {
		  subgraphs[current_partition]->dst[j++] = graph->dst[k++] - current_offset_dst;
		  subgraphs[current_partition]->num_edges++;
		}
	  }
	  /*
	  printf("\n-----------------------\n");
	  printf("Current Partition: %d\n", current_partition);
	  printCSR(subgraphs[0]);
	  printf("\n");
	  printCSR(subgraphs[1]);
	  */
	}
	return subgraphs;
   
}
/*
int main(int argc, char* argv[]){
  if (argc < 2) {
    printf("No arguments, enter number of rows in graph\n");
    exit(1);
  }
  int rows = atoi(argv[1]);
  CSR_Graph* graph = create_grid_like_csr_graph(rows);
  printCSR(graph);
  printf("\n");
  int num = 2;
  CSR_Graph** graphs = partition_graph(graph, num);
  for (int i = 0; i < num; i++) {
	printCSR(graphs[i]);
	printf("\n");
  }
  dealloc_csr_graph(graph);
  for (int i = 0; i < num; i++) dealloc_csr_graph(graphs[i]);
  free(graphs);
  exit(0);
}

*/
