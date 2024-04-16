#include "BFS_serial_c.h"
#include "BFS_Cuda.cu"
#include <time.h>
int main(int argc, char* argv[]){
  if (argc < 2) {
    printf("No arguments, enter number of rows in graph\n");
    exit(1);
  }
  int rows = atoi(argv[1]);
  clock_t start = clock();
  CSR_Graph* graph = create_grid_like_csr_graph(rows);
  printCSR(graph);
  //Graph* graph1 = create_grid_like_graph(rows);
  //int found = serial_BFS(graph1, 0, rows * rows - 1);
  //int found1 = serial_csr_BFS(graph, 0);
  //printf("found? %d\n", found);
  //printf("found? %d\n", found1);
  clock_t stop = clock();
  double duration = (double)(stop - start) / CLOCKS_PER_SEC;
  printf("Serial C took %f seconds\n", duration);
  start = clock();
  execute_bfs(graph);
  stop = clock();
  duration = (double)(stop - start) / CLOCKS_PER_SEC;
  printf("Cuda took %f seconds\n", duration);
  dealloc_csr_graph(graph);
  //dealloc_graph(graph1);
  exit(0);
}
