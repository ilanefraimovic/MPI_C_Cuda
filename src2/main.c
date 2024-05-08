#include "../src/BFS_serial_c.h"
#include "../src/partition.h"
#include "../src/graph_create.h"

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
  CSR_Graph** graphs;
  CSR_Graph* sub_graph = (CSR_Graph*)malloc(sizeof(CSR_Graph));
  CSR_Graph* graph;
  clock_t start, stop;
  double mpi_duration;

  if (pid == MASTER) {
	int rows = atoi(argv[1]);
	graph = create_grid_like_csr_graph(rows);
	printCSR(graph);
    start = clock();
	printf("1111\n");

	final_node = graph->num_vertices - 1;
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
  int found = MPI_execute_BFS(sub_graph,0, final_node, pid);
  dealloc_csr_graph(sub_graph);
  int *mpi_results = NULL:
  if (pid == MASTER) {
	mpi_results = (int*)malloc(sizeof(int) * nprocs);
	if (mpi_results == NULL) {
	  printf("Memory Alloc Failed\n");
	  exit(1);
	}
  }
  MPI_Gather(&found, 1, MPI_FLOAT, mpi_results, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (pid == MASTER) {
	int found = 0;
	for (int i = 0; i < nprocs; i++) if (mpi_results[i]) found = 1;
	stop = clock();
	mpi_duration = (double)(stop - start) / CLOCKS_PER_SEC;
	printf("MPI succeeded? %d\n", found);
	printf("MPI took %f seconds\n", mpi_duration);
	dealloc_csr_graph(graph);
	free(mpi_results);
	free(graphs);

  }
  
  MPI_Finalize();
  exit(0);
}
