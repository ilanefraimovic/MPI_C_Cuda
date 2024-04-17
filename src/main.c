#include "BFS_serial_c.h"
#include "BFS_Cuda.cuh"
#include "graph_create.h"
#include <time.h>
int main(int argc, char* argv[]){
  if (argc < 2) {
    printf("No arguments, enter number of rows in graph\n");
    exit(1);
  }
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
  
  printf("Final Results:\n");
  printf("Graph Size: %d\n", graph->num_vertices);
  printf("Time it took serial c: %f\n", c_duration);
  printf("Time it took cuda    : %f\n", cuda_duration);
  dealloc_csr_graph(graph);
  
  exit(0);
}
