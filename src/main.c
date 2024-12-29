#include "BFS_serial_c.h"
#include "BFS_Cuda.cuh"
#include "BFS_MPI_Cuda.cuh"
#include "partition.h"
#include "graph_create.h"

#include <time.h>
#include "mpi.h"

#define MASTER 0

int main(int argc, char* argv[]){
  if (argc < 2) {
    printf("No arguments, enter number of rows in graph\n");
    exit(1);
  }
  int nprocs, pid, name_len;
  unsigned int final_node;
  char proc_name[MPI_MAX_PROCESSOR_NAME];
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  MPI_Get_processor_name(proc_name, &name_len);
  printf("Hello from processor %s\n", proc_name);
  int num_devices;
  cudaGetDeviceCount(&num_devices);
  printf("Cuda Devices: %d\n", num_devices);
  CSR_Graph** graphs;
  CSR_Graph* sub_graph = (CSR_Graph*)malloc(sizeof(CSR_Graph));
  CSR_Graph* graph;
  clock_t start, stop;
  double mpi_cuda_duration, c_duration, cuda_duration;

  if (pid == MASTER) {
	int rows = atoi(argv[1]);
	graph = create_grid_like_csr_graph(rows);
	printCSR(graph);
    start = clock();
	printf("1111\n");
	int found = serial_csr_BFS(graph, 0);
	printf("2222\n");
    stop = clock();
	c_duration = (double)(stop - start) / CLOCKS_PER_SEC;
	printf("Serial C succeeded? %d\n", found);
	printf("Serial C took %f seconds\n", c_duration);
	
	start = clock();
	found = cuda_execute_bfs(graph);
	stop = clock();
    cuda_duration = (double)(stop - start) / CLOCKS_PER_SEC;
	printf("Cuda succeeded? %d\n", found);
	printf("Cuda took %f seconds\n", cuda_duration);

	final_node = graph->num_vertices - 1;

	start = clock();
    graphs = partition_graph(graph, nprocs);
	for (int i = 1; i < nprocs; i++) {
	  MPI_Send(&(graphs[i]->num_vertices), 1, MPI_UNSIGNED, i, 0, MPI_COMM_WORLD);
	  printf("hello\n");
	  MPI_Send(&(graphs[i]->num_edges), 1, MPI_UNSIGNED, i, 0, MPI_COMM_WORLD);
	  printf("hello\n");
	  MPI_Send(&(graphs[i]->offset), 1, MPI_UNSIGNED, i, 0, MPI_COMM_WORLD);
	  printf("hello\n");
	  MPI_Send(graphs[i]->rowPtrs, graphs[i]->num_vertices, MPI_UNSIGNED, i, 0, MPI_COMM_WORLD);
	  printf("hello\n");
	  MPI_Send(graphs[i]->dst, graphs[i]->num_edges, MPI_UNSIGNED, i, 0, MPI_COMM_WORLD);
	  printf("hello\n");
	}
	sub_graph = graphs[0];
	for (int i = 0; i < nprocs; i++) printCSR(graphs[i]);
    for (int i = 1; i < nprocs; i++) dealloc_csr_graph(graphs[i]);
  }
  MPI_Bcast(&final_node, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
  if (pid != MASTER) {
	MPI_Recv(&(sub_graph->num_vertices), 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&(sub_graph->num_edges), 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	MPI_Recv(&(sub_graph->offset), 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    sub_graph->rowPtrs = (unsigned int*)malloc((sub_graph->num_vertices) * sizeof(unsigned int));
    MPI_Recv(sub_graph->rowPtrs, sub_graph->num_vertices, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    sub_graph->dst = (unsigned int*)malloc(sub_graph->num_edges * sizeof(unsigned int));
    MPI_Recv(sub_graph->dst, sub_graph->num_edges, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  int found = mpi_cuda_execute_bfs(sub_graph, final_node, pid);
  dealloc_csr_graph(sub_graph);
  int *cuda_mpi_results = NULL;
  if (pid == MASTER) {
	cuda_mpi_results = (int*)malloc(sizeof(int) * nprocs);
	if (cuda_mpi_results == NULL) {
	  printf("Memory Alloc Failed\n");
	  exit(1);
	}
  }
  MPI_Gather(&found, 1, MPI_FLOAT, cuda_mpi_results, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (pid == MASTER) {
	int found = 0;
	for (int i = 0; i < nprocs; i++) if (cuda_mpi_results[i]) found = 1;
	stop = clock();
	mpi_cuda_duration = (double)(stop - start) / CLOCKS_PER_SEC;
	printf("Cuda + MPI succeeded? %d\n", found);
	printf("Cuda + MPI took %f seconds\n", mpi_cuda_duration);
  }
  
  if (pid == MASTER) {
	printf("Final Results:\n");
	printf("Graph Size: %d\n", graph->num_vertices);
	printf("Time it took serial c: %f\n", c_duration);
	printf("Time it took cuda    : %f\n", cuda_duration);
	printf("Time it took cuda+MPI: %f\n", mpi_cuda_duration);
	dealloc_csr_graph(graph);
	free(cuda_mpi_results);
	//dealloc_csr_graph(graphs[0]);
	free(graphs);
  }
  MPI_Finalize();
  exit(0);
}
