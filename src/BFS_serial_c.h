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

int serial_csr_BFS(CSR_Graph* graph, int source) {
  int visited[MAX];
  int frontier[MAX];
  int front = 0, back = 0;
  for(int i = 0; i < graph->num_vertices; i++) visited[i] = 0;
  frontier[0] = source;
  back++;
  visited[0] = 1;
  while (front < back) {
	int currentVertex = frontier[front++];
	printf("Visited: %d\n", currentVertex);

	for (int i = graph->rowPtrs[currentVertex]; i < graph->rowPtrs[currentVertex + 1]; i++) {
	  int adjVertex =  graph->dst[i];
	  if (!visited[adjVertex]) {
		visited[adjVertex] = 1;
		frontier[back++] = adjVertex;
	  }
	}
  }
  return 0;
}

int serial_BFS(Graph* graph, int source, int dest) {
  int visited[MAX];
  int frontier[MAX];
  int front = 0, back = 0;
  for(int i = 0; i < graph->num; i++)
    visited[i] = 0;
  frontier[0] = source;
  back++;
  visited[0] = 1;
  while(front != back) {
    source = frontier[front++];
    printf("Visited: %d\n", source);
    if (source == dest) return 1;
	
    for(int i = 0; i < graph->num; i++) {
      if (graph->adj[source][i] && !visited[i]) {
		frontier[back++] = i;
		visited[i] = 1;
      }
    }
  }
  return 0;
}

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
//src = [0, 2, 4, 5, 6, 8, 10, 11, 12 ]
//dst = [1, 3, 2, 4, 5, 4, 6, 5, 7, 8, 7, 8]
