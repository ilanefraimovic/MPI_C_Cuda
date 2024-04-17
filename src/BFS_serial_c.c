#include "BFS_serial_c.h"

int serial_csr_BFS(CSR_Graph* graph, int source) {
  int* visited = (int*)malloc(graph->num_vertices * sizeof(int));
  int* frontier = (int*)malloc(graph->num_vertices * sizeof(int));
  if (visited == NULL || frontier == NULL) {
	printf("Unable to allocated memory in serial BFS\n");
	exit(1);
  }
  int front = 0, back = 0;
  for(unsigned int i = 0; i < graph->num_vertices; i++) visited[i] = 0;
  frontier[0] = source;
  back++;
  visited[0] = 1;
  while (front < back) {
	unsigned int currentVertex = frontier[front++];
	printf("Visited: %d\n", currentVertex);
	if (currentVertex == graph->num_vertices - 1) {
	  free(visited);
	  free(frontier);
	  return 1;
	}
	for (unsigned int i = graph->rowPtrs[currentVertex]; i < graph->rowPtrs[currentVertex + 1]; i++) {
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
