#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define MAX 78000
typedef struct Graph {
  //# of vertices
  int num;
  //adjacency matrix
  int adj[MAX][MAX];
} Graph;

typedef struct CSR_Graph {
  int num_vertices;
  int num_edges;
  int rowPtrs[MAX];
  int dst[MAX];
  int last_row_ptr;
} CSR_Graph;

CSR_Graph* alloc_csr_graph(int vertices) {
  CSR_Graph* graph = (CSR_Graph*) malloc(sizeof(CSR_Graph));
  if (graph == NULL) {
    printf("Memory Alloc Failed\n");
    exit(1);
  }
  graph->num_vertices = vertices;
  for (int i = 0; i < vertices; i++) {
	graph->rowPtrs[i] = 0;
	graph->dst[i] = 0;
  }
  return graph;
}

void dealloc_csr_graph(CSR_Graph* graph) {
  free(graph);
}

void add_csr_edge(CSR_Graph* graph, int x) {
  graph->dst[graph->num_edges++] = x;
}

CSR_Graph* create_grid_like_csr_graph(int x) {
  CSR_Graph* graph = alloc_csr_graph(x*x);
  graph->num_vertices = x*x;
  graph->num_edges = 0;
  graph->last_row_ptr = 0;
  for (int i = 0; i < x*x; i++){
    if ((i + 1) % x != 0) add_csr_edge(graph, i+1);
    if (i < x * (x - 1)) add_csr_edge(graph, i+x);
	graph->rowPtrs[++graph->last_row_ptr] = graph->num_edges;
  }
  return graph;
}

void printCSR(CSR_Graph* graph) {
  printf("Num vertices: %d\n", graph->num_vertices);
    printf("RowPtrs: ");
    for (int i = 0; i <= graph->last_row_ptr; i++) {
        printf("%d ", graph->rowPtrs[i]);
    }
    printf("\nDestination Indices: ");
    for (int i = 0; i < graph->num_vertices; i++) {
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

