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

/*
  0 1 2 3 4 5 6 7 8
0   1   1
1     1   1
2           1
3         1   1
4           1   1  
5                 1
6               1 
7                 1
8

0->1->2
|  |  |
3->4->5
|  |  |
6->7->8
*/
//src = [0, 2, 4, 5, 7,             9, 10, 11, 12]
//dst = [1, 3, 2, 4, 5, 4, 6, 5, 7, 8, 7, 8]

//src = [0, 2, 4, 5, 7]
//dst = [1, 3, 2, 4, 5, 4, 6, 5, 7]

// 9, 10, 11, 12
// 8, 7, 8

