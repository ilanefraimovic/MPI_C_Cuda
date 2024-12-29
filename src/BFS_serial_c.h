#ifndef BFS_SERIAL_H
#define BFS_SERIAL_H
#include "graph_create.h"

int serial_csr_BFS(CSR_Graph* graph, int source);
int MPI_execute_BFS(CSR_Graph* graph, int source, int dest, int pid);
int serial_BFS(Graph* graph, int source, int dest);

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
//src = [0, 2, 4, 5, 6, 8, 10, 11, 12 ]
//dst = [1, 3, 2, 4, 5, 4, 6, 5, 7, 8, 7, 8]
