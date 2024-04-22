#include "BFS_MPI_Cuda.cuh"
int execute_bfs(CSR_Graph* graph) {
  
}

void partition_graph(CSRGraph* graph, CSRGraph* subgraphs, int world_size) {
  //check if the number of partitions is valid
    if (num_partitions <= 1) {
        printf("Number of partitions should be greater than 1.\n");
        return NULL;
    }

    //allocate memory for array of partitioned graphs
    subgraphs = (CSR_Graph*)malloc(num_partitions * sizeof(CSR_Graph));
    if (subgraphs == NULL) {
        printf("Memory allocation failed.\n");
        return NULL;
    }

    //calculate the number of vertices per partition

	unsigned int vertices_per_partition = graph->num_vertices / num_partitions;

	//initialize each partitioned graph
    for (unsigned int i = 0; i < num_partitions; ++i) {
        subgraphs[i].num_vertices = vertices_per_partition;
        subgraphs[i].rowPtrs = (unsigned int*)malloc((vertices_per_partition + 1) * sizeof(unsigned int));
        subgraphs[i].num_edges = 0;
        subgraphs[i].dst = (unsigned int*)malloc((vertices_per_partition * 2) * sizeof(unsigned int));
        subgraphs[i].last_row_ptr = 0;
    }

	//partitioning logic
	for (int i = 0, j = 0, k = 0; i < graph->num_vertices; i++, j++, k++) {
	  current_partition = i / vertices_per_partition;
  
	  if (j == vertices_per_partition) {	
		while (k < graph->rowPtrs[i]) {
		  subgraphs[current_partition - 1]->dst[j++] = graph->dst[k++];
		}
		subgraphs[current_partition - 1]->num_edges = k;
		j = 0;
	  }
	  
	  subgraphs[current_partition]->RowPtrs[j] = graph->rowPtrs[i];
	  subgraphs[current_partition]->dst[j] = graph->dst[k];
	}
   
    return graphs;
}

int main(int argc, char* argv[]){
  if (argc < 2) {
    printf("No arguments, enter number of rows in graph\n");
    exit(1);
  }
  int rows = atoi(argv[1]);
  CSR_Graph* graph = create_grid_like_csr_graph(rows);
  printCSR(graph);
  CSR_Graph* graphs;
  partition_graph(graph, graphs, 2);
  printCSR(graphs[0]);
  printCSR(graphs[2]);
  dealloc_csr_graph(graph);
  
  exit(0);
}
