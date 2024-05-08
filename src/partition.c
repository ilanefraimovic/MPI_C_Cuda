#include "partition.h"

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
	unsigned int vertices_per_partition = (graph->num_vertices / num_partitions);
	while (vertices_per_partition % (int)sqrt((double)graph->num_vertices) != 0) vertices_per_partition++; 
	//initialize each partitioned graph
    for (int i = 0; i < num_partitions; i++) {
	    subgraphs[i] = (CSR_Graph*)malloc(sizeof(CSR_Graph)); // allocate memory for CSR_Graph
		if (subgraphs[i] == NULL) {
		  printf("Memory allocation failed.\n");
		  exit(1);
		}
        subgraphs[i]->num_vertices = 0;
        subgraphs[i]->rowPtrs = (unsigned int*)malloc((vertices_per_partition + 1) * sizeof(unsigned int));
        subgraphs[i]->num_edges = 0;
        subgraphs[i]->dst = (unsigned int*)malloc((vertices_per_partition * 2) * sizeof(unsigned int));
        subgraphs[i]->last_row_ptr = 0;
		subgraphs[i]->offset = 0;
    }
	

	//partitioning logic
	int current_offset_src = 0;
	int current_offset_dst = 0;
	int last_offset_dst = 0;
	for (unsigned int i = 0, j = 0, k = 0; i < graph->num_vertices; i++, j++, k++) {
	  int current_partition = i / vertices_per_partition;
	  if (j == vertices_per_partition) {
		last_offset_dst = current_offset_dst;
		current_offset_dst += vertices_per_partition;
		while (k < graph->rowPtrs[i]) {
		  subgraphs[current_partition - 1]->dst[j++] = graph->dst[k++] - last_offset_dst;
		  subgraphs[current_partition - 1]->num_edges++;
		}
		current_offset_src = graph->rowPtrs[i];
		subgraphs[current_partition]->offset = current_offset_dst;
		j = 0;
	  }
	  
	  subgraphs[current_partition]->rowPtrs[j] = graph->rowPtrs[i] - current_offset_src;
	  subgraphs[current_partition]->num_vertices++;
	  subgraphs[current_partition]->dst[j] = graph->dst[k] - current_offset_dst;
	  subgraphs[current_partition]->num_edges++;
	  if (i == graph->num_vertices - 1) {
		while (k < graph->num_edges) {
		  subgraphs[current_partition]->dst[j++] = graph->dst[k++] - current_offset_dst;
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
