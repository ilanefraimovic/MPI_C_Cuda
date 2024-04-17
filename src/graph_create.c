#include "graph_create.h"
CSR_Graph* alloc_csr_graph(int vertices) {
  printf("Hello from beginning of alloc_csr_graph\n");
  CSR_Graph* graph = (CSR_Graph*) malloc(sizeof(CSR_Graph));
  if (graph == NULL) {
    printf("Memory Alloc Failed\n");
    exit(1);
  }
  graph->num_vertices = vertices;
  graph->rowPtrs = (unsigned int*)malloc((graph->num_vertices + 1) * sizeof(unsigned int));
  graph->dst = (unsigned int*)malloc(2 * graph->num_vertices * sizeof(unsigned int));
  if (graph->rowPtrs == NULL || graph->dst == NULL) {
        printf("Memory Allocation Failed\n");
        exit(1);
  }
  for (int i = 0; i < vertices; i++) {
	graph->rowPtrs[i] = 0;
	graph->dst[i] = 0;
	graph->dst[i + vertices] = 0;
  }
  printf("Hello from end of alloc_csr_graph\n");
  return graph;
}

void dealloc_csr_graph(CSR_Graph* graph) {
  free(graph->rowPtrs);
  free(graph->dst);
  free(graph);
}

void add_csr_edge(CSR_Graph* graph, unsigned int x) {
  graph->dst[graph->num_edges++] = x;
}

CSR_Graph* create_grid_like_csr_graph(unsigned int x) {
  printf("Hello from beginning of create_grid_like_csr_graph\n");
  CSR_Graph* graph = alloc_csr_graph(x*x);
  graph->num_edges = 0;
  graph->last_row_ptr = 0;
  for (unsigned int i = 0; i < x*x; i++){
    if ((i + 1) % x != 0) add_csr_edge(graph, i+1);
    if (i < x * (x - 1)) add_csr_edge(graph, i+x);
	graph->rowPtrs[++graph->last_row_ptr] = graph->num_edges;
  }
  printf("Hello from end of create_grid_like_csr_graph\n");
  return graph;
}

void printCSR(CSR_Graph* graph) {
  printf("Num vertices: %d\n", graph->num_vertices);
    printf("RowPtrs: ");
    for (unsigned int i = 0; i <= graph->last_row_ptr; i++) {
        printf("%d ", graph->rowPtrs[i]);
    }
    printf("\nDestination Indices: ");
    for (unsigned int i = 0; i < graph->num_vertices; i++) {
        printf("%d ", graph->dst[i]);
    }
    printf("\n");
}

Graph* alloc_graph(int vertices) {
  Graph* graph = (Graph*) malloc(sizeof(Graph));
  if (graph == NULL) {
    printf("Memory Alloc Failed\n");
    exit(1);
  }
  graph->num = vertices;
  for(int i = 0; i < vertices; i++)
    for(int j = 0; j < vertices; j++)
      graph->adj[i][j] = 0;
  return graph;
}

void dealloc_graph(Graph* graph) {
  free(graph);
}

void add_edge(Graph* graph, int x, int y) {
  graph->adj[x][y] = 1;
}

Graph* create_grid_like_graph(int x) {
  Graph* graph = alloc_graph(x*x);
  for (int i = 0; i < x*x; i++){
    if ((i + 1) % x != 0)
      add_edge(graph, i, i+1);
    if (i < x * (x - 1))
      add_edge(graph, i, i+x);
  }
  return graph;
}
