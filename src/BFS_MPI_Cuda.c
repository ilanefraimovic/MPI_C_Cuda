#include "graph_create.h"
int execute_bfs(CSR_Graph* graph) {
  
}

CSR_Graph** partition_graph(CSR_Graph* graph, int num_partitions) {
  //check if the number of partitions is valid
    if (num_partitions <= 1) {
        printf("Number of partitions should be greater than 1.\n");
        exit(1);
    }

    //allocate memory for array of partitioned graphs
    CSR_Graph** subgraphs = (CSR_Graph**)malloc(num_partitions * sizeof(CSR_Graph));
    if (subgraphs == NULL) {
        printf("Memory allocation failed.\n");
        exit(1);
    }

    //calculate the number of vertices per partition
	unsigned int vertices_per_partition = (graph->num_vertices / num_partitions) + 1;
	//initialize each partitioned graph
    for (int i = 0; i < num_partitions; i++) {
	    subgraphs[i] = (CSR_Graph*)malloc(sizeof(CSR_Graph)); // allocate memory for CSR_Graph
		if (subgraphs[i] == NULL) {
		  printf("Memory allocation failed.\n");
		  exit(1);
		}
        subgraphs[i]->num_vertices = vertices_per_partition;
        subgraphs[i]->rowPtrs = (unsigned int*)malloc((vertices_per_partition + 1) * sizeof(unsigned int));
        subgraphs[i]->num_edges = 0;
        subgraphs[i]->dst = (unsigned int*)malloc((vertices_per_partition * 2) * sizeof(unsigned int));
        subgraphs[i]->last_row_ptr = 0;
    }
	

	//partitioning logic
	for (int i = 0, j = 0, k = 0; i < graph->num_vertices; i++, j++, k++) {
	  int current_partition = i / vertices_per_partition;
	  if (j == vertices_per_partition) {	
		while (k < graph->rowPtrs[i]) {
		  subgraphs[current_partition - 1]->dst[j++] = graph->dst[k++];
		  subgraphs[current_partition - 1]->num_edges++;
		}
		j = 0;
	  }
	  
	  subgraphs[current_partition]->rowPtrs[j] = graph->rowPtrs[i];
	  subgraphs[current_partition]->dst[j] = graph->dst[k];
	  subgraphs[current_partition]->num_edges++;
	  if (i == graph->num_vertices - 1) {
		while (k < graph->num_edges) {
		  subgraphs[current_partition]->dst[j++] = graph->dst[k++];
		  subgraphs[current_partition]->num_edges++;
		}
	  }
	  /*
	  printf("\n-----------------------\n");
	  printf("Current Partition: %d\n", current_partition);
	  printCSR(subgraphs[0]);
	  printf("\n");
	  printCSR(subgraphs[1]);
	  */
	}
	return subgraphs;
   
}

int main(int argc, char* argv[]){
  if (argc < 2) {
    printf("No arguments, enter number of rows in graph\n");
    exit(1);
  }
  int rows = atoi(argv[1]);
  CSR_Graph* graph = create_grid_like_csr_graph(rows);
  printCSR(graph);
  printf("\n");
  int num = 2;
  CSR_Graph** graphs = partition_graph(graph, num);
  for (int i = 0; i < num; i++) {
	printCSR(graphs[i]);
	printf("\n");
  }
  dealloc_csr_graph(graph);
  for (int i = 0; i < num; i++) dealloc_csr_graph(graphs[i]);
  free(graphs);
  exit(0);
}
