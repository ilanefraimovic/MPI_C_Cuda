
#include <stdio.h>
#include <mpi.h>
#include "get_ip.h"
#define MASTER 0
int main(int argc, char *argv[]) {
	int nprocs, pid, namelength, i, j, n;
	float a=1.5, b=2.9, c=0.89;
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	char *ip = NULL;

	get_ip(&ip); printf("MASTER: ip=%s\n", ip);
	n=3;
	
	if (argc == 2) n = atoi(argv[1]);
	MPI_Init (&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &pid);
	MPI_Get_processor_name(processor_name, &namelength);

	for (i=0; i<n;i++) {
		if (pid == MASTER) {
			printf("%s [%d/%d]: master %d: Hello! %d/%d\n", ip, pid, nprocs, pid, i, n);
		}
		else if (pid == 1) {
			printf("%s [%d/%d]: master %d: World! %d/%d\n", ip, pid, nprocs, pid, i, n);
		}
		else {
			printf("%s [%d/%d]: master %d: Nothing to say! %d/%d\n", ip, pid, nprocs, pid, i, n);
		}
	}
	MPI_Finalize();
}

