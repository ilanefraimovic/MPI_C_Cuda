#include "BFS_serial_c.h"
#include "BFS_Cuda.cuh"
#include "BFS_MPI_Cuda.c"
#include "graph_create.h"

#include <time.h>
#include "mpi.h"

#define MASTER 0

int main(int argc, char* argv[]){
  if (argc < 2) {
    printf("No arguments, enter number of rows in graph\n");
    exit(1);
  }
  int nprocs, pid, namelength;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  MPI_Get_processor_name(processor_name, &namelength);
  if (pid == MASTER) {
	int rows = atoi(argv[1]);
	CSR_Graph* graph = create_grid_like_csr_graph(rows);
	//printCSR(graph);
	clock_t start = clock();
	printf("1111\n");
	int found = serial_csr_BFS(graph, 0);
	printf("2222\n");
	clock_t stop = clock();
	double c_duration = (double)(stop - start) / CLOCKS_PER_SEC;
	printf("Serial C succeeded? %d\n", found);
	printf("Serial C took %f seconds\n", c_duration);
	
	start = clock();
	found = execute_bfs(graph);
	stop = clock();
	double cuda_duration = (double)(stop - start) / CLOCKS_PER_SEC;
	printf("Cuda succeeded? %d\n", found);
	printf("Cuda took %f seconds\n", cuda_duration);

	start = clock();
    CSR_Graph** graphs = partition_graph(graph, nprocs);
	for (int i = 1; i < nprocs; i++) {
	  MPI_Send(&(graphs[i]->num_vertices), 1, MPI_UNSIGNED, i, 0, MPI_COMM_WORLD);
	  MPI_Send(&(graphs[i]->num_edges), 1, MPI_UNSIGNED, i, 0, MPI_COMM_WORLD);
	  MPI_Send(graphs[i]->rowPtrs, graphs[i]->num_vertices + 1, MPI_UNSIGNED, i, 0, MPI_COMM_WORLD);
	  MPI_Send(graphs[i]->num_vertices, graphs[i]->num_edges + 1, MPI_UNSIGNED, i, 0, MPI_COMM_WORLD);
	}
  }
  CSR_Graph* sub_graph = (CSR_Graph*)malloc(sizeof(CSR_Graph));
  if (pid != MASTER) {
	MPI_Recv(&(sub_graph->num_vertices), 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&(sub_graph->num_edges), 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    sub_graph->rowPtrs = (unsigned int*)malloc((sub_graph->num_vertices + 1) * sizeof(unsigned int));
    MPI_Recv(sub_graph->rowPtrs, sub_graph->num_vertices + 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    sub_graph->dst = (unsigned int*)malloc(sub_graph->num_edges * sizeof(unsigned int));
    MPI_Recv(sub_graph->dst, sub_graph->num_edges + 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  } else {
	sub_graph = graphs[0];
  }
  found = execute_bfs(sub_graph);
  int *cuda_mpi_results = NULL;
  if (pid == MASTER) {
	cuda_mpi_results = malloc(sizeof(int) * nprocs);
	if (graph == NULL) {
	  printf("Memory Alloc Failed\n");
	  exit(1);
	}
  }
  MPI_Gather(&found, 1, MPI_FLOAT, cuda_mpi_results, 1, MPI_INT, 0, MPI_COMM_WORLD);
  
  if (pid == MASTER) {
	found = 0;
	for (int i = 0; i < nprocs; i++) {
	  if (cuda_mpi_results[i]) found = 1;
	}
	stop = clock();
	double mpi_cuda_duration = (double)(stop - start) / CLOCKS_PER_SEC;
	printf("Cuda + MPI succeeded? %d\n", found);
	printf("Cuda + MPI took %f seconds\n", mpi_cuda_duration);
	
	printf("Final Results:\n");
	printf("Graph Size: %d\n", graph->num_vertices);
	printf("Time it took serial c: %f\n", c_duration);
	printf("Time it took cuda    : %f\n", cuda_duration);
	printf("Time it took cuda+MPI: %f\n", cuda_duration);
	dealloc_csr_graph(graph);
	for (int i = 0; i < num; i++) dealloc_csr_graph(graphs[i]);
	free(graphs);
	free(cuda_mpi_results);
  }
  dealloc_csr_graph(sub_graph);
  exit(0);
}
