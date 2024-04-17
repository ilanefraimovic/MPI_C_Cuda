#ifndef GRAPH_CREATE_H
#define GRAPH_CREATE_H

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
  unsigned int num_vertices;
  unsigned int num_edges;
  unsigned int* rowPtrs;
  unsigned int* dst;
  unsigned int last_row_ptr;
} CSR_Graph;

CSR_Graph* alloc_csr_graph(int vertices);
void dealloc_csr_graph(CSR_Graph* graph);
void add_csr_edge(CSR_Graph* graph, unsigned int x);
CSR_Graph* create_grid_like_csr_graph(unsigned int x);
void printCSR(CSR_Graph* graph);

Graph* alloc_graph(int vertices);
void dealloc_graph(Graph* graph);
void add_edge(Graph* graph, int x, int y);
Graph* create_grid_like_graph(int x);

#endif

