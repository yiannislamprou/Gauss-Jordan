// GJ_KJI.c 
// Ioannis Lamprou
// Gauss-Jordan (KJI) implementation via MPI

#include <stdio.h>
#include <stdlib.h>
#include <math.h> // compile with -lm
#include <mpi.h>  // compile using mpicc

void pivoting(int, double**, int*, int);
void modify(int, int, double**, int*, int);

int main(int argc, char **argv){
	// argv[1] : dimension n
	// argv[2] : random seed to generate [A|b]
	// argv[3] : maximum element in the matrix
	// argv[4] : blocksize (if == n/p then block, otherwise blocksize-shuffle)

	int numprocs, id, k, i, j, n, l, *owner, *piv, *dims, ndims, *periods, blocksize, dest, source, seed, upper;
	double **A, *colBuf, start, end;
	MPI_Comm HyperCube;
	MPI_File fh;
	MPI_Status status;

	// Check and Keep arguments
	sscanf(argv[1], "%d", &n);
	sscanf(argv[2], "%d", &seed);
	sscanf(argv[3], "%d", &upper);
	sscanf(argv[4], "%d", &blocksize);

	// Initialize MPI
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD,&numprocs); 

	// Create cube of size p
	ndims = (int) log2(numprocs);
	dims = malloc(ndims*sizeof(int));
	if (dims == NULL){
		fprintf(stderr, "Memory allocation failed\n");
		MPI_Finalize();
		return -1;
	}	
	periods = malloc(ndims*sizeof(int)); 
	if (periods == NULL){
		fprintf(stderr, "Memory allocation failed\n");
		MPI_Finalize();
		return -1;
	}
	for(i = 0; i < ndims; i++){
		dims[i] = 2;
		periods[i] = 0;
	}	
	MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, 1, &HyperCube);
	MPI_Comm_rank(HyperCube,&id);
	printf("Process %d is ready\n", id); 

	// Create Matrix [A|b]
	A = malloc(n*sizeof(double*));
	if (A == NULL){
		fprintf(stderr, "Memory allocation failed\n");
		MPI_Finalize();
		return -1;
	}	
	for(i = 0; i < n; i++){
		A[i] = malloc((n+1)*sizeof(double));
		if (A[i] == NULL){
			fprintf(stderr, "Memory allocation failed\n");
			MPI_Finalize();
			return -1;
		}
	}		

	srand(seed);
	for(i = 0; i < n; i++)
		for(j = 0; j < n+1; j++){
				A[i][j] = rand() % upper + 1;
			}	

	//for(i = 0; i < n; i++){
	//	for(j = 0; j < n+1; j++)
	//		printf("%f ", A[i][j]);
	//	printf("\n");
	//}	

	// Define an owner for each column
	owner = malloc((n+1)*sizeof(int));
	if (owner == NULL){
		fprintf(stderr, "Memory allocation failed\n");
		MPI_Finalize();
		return -1;
	}	

	for(j = 0; j < n; j++)
			owner[j] = (j/blocksize) % numprocs;
			// blocksize == n/numprocs means block (extreme case of blocksize-shuffling)
	owner[n] = 0;

	// Assigning memory to send-receive buffer
	colBuf = malloc((n+1)*sizeof(double));	// column + pivot
	if (colBuf == NULL){
		fprintf(stderr, "Memory allocation failed\n");
		MPI_Finalize();
		return -1;
	}	
	piv = malloc(n*sizeof(int));
	if (piv == NULL){
		fprintf(stderr, "Memory allocation failed\n");
		MPI_Finalize();
		return -1;
	}	

	// Main Algorithm (time performance)
	MPI_Barrier(HyperCube);	// all processes have completed the pre-processing
	start = MPI_Wtime(); // capturing starting time

	for(k = 0; k < n; k++)
		piv[k] = k;

	for(k = 0; k < n; k++){
		if(owner[k] == id){
			pivoting(k, A, piv, n);
			// Update column buffer
			for(i = 0; i < n; i++)
				colBuf[i] = A[i][k];
			colBuf[n] = (double) piv[k];
		}
		// SEND/RECEIVE COLUMN
		// Use of tree
		for(l = 1; l < numprocs; l *= 2){
			if((id^owner[k]) < l){
				dest = (id^owner[k]) + l;
				if(dest < numprocs){
					// blocking send
					MPI_Send(colBuf, n+1, MPI_DOUBLE, dest^owner[k], 0, HyperCube);// err
				}
			}
			else if((id^owner[k]) < 2*l){
				source = (id^owner[k]) - l;
				// blocking receive
				MPI_Recv(colBuf, n+1, MPI_DOUBLE, source^owner[k], 0, HyperCube, &status);//err
				// update A and piv
				for(i = 0; i < n; i++)
					A[i][k] = colBuf[i];
				piv[k] = (int) colBuf[n];	    
			}
		}
		// update the rest of the columns
		for(j = k+1; j < n+1; j++) // n+1 to account for b
			if(owner[j] == id)
				modify(k, j, A, piv, n);
	}

	MPI_Barrier(HyperCube); // all processes have completed in parallel
	end = MPI_Wtime();	// now (end - start) holds the max parallel time (due to the barrier before)

	// The owner of the last column prints the result and the elapsed time
	if(owner[n] == id){
	printf("Computations completed in %f seconds...\n A = \n", end - start);
	// Print Matrix (or maybe just the result)
	//	for(i = 0; i < n; i++){
	//		for(j = 0; j < n+1; j++)
	//			printf("%3.3f ", A[i][j]);
	//		printf("\n");
	//	}		
	}

	MPI_Finalize();

	// free all memory used
	free(owner);
	free(piv);
	free(dims);
	free(periods);
	free(colBuf);
	for(i = 0; i < n; i++)
		free(A[i]);
	free(A);	

	return 0;
}

// Functions for pivoting and modifying columns

void pivoting(int k, double **A, int *piv, int n){
	double max, temp;
	int argmax, l;
	
	// find max guide
	max = A[k][k];
	argmax = k;
	for(l = k+1; l < n; l++){
		if (A[l][k] > max){
			max = A[l][k];
			argmax = l;
		}	
	}
	if(k != argmax){
		// fix pivot
		piv[k] = argmax;
		// swap elements
		if(argmax != k){
			temp = A[piv[k]][k];
			A[piv[k]][k] = A[k][k];
			A[k][k] = temp;
		}
	}	
	
	return;
}	
	
void modify(int k, int j, double **A, int *piv, int n){
	double temp;
	int i;

	// swap elements
	temp = A[piv[k]][j];
	A[piv[k]][j] = A[k][j];
	A[k][j] = temp;
	// keep multiplier
	A[k][j] = A[k][j]/A[k][k];
	// update column
	for(i = 0; i < n; i++)
		if(i != k)
			A[i][j] = A[i][j] - A[i][k]*A[k][j];
			
	return;
}	
			
// end of file
