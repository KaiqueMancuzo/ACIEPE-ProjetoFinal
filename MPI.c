#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>
#include <math.h>

#define CONV_THRESHOLD 1.0e-5f // threshold of convergence

// matrix to be solved
double **grid;

// auxiliary matrix
double **new_grid;

// size of each side of the grid
int size;
int iter_max_num;

// return the absolute value of a number
double absolute(double num){
    if(num < 0)
        return -1.0 * num;
    return num;
}

int main(int argc, char *argv[]){

    //variaveis referentes às comunicações do mpi
    int rank, num_procs;
    int tag_top = 0, tag_bottom = 1;
    MPI_Status status;

    //número de iterações e verificação de convergencia
    int iter = 0;
    int hasError = 1;
    double diffnorm, gdiffnorm;

    struct timeval time_start, time_end;

    //inicializar os processos
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if(rank == 0){
      if (fscanf(stdin, "%d", &size) != 1) {
        printf("Error: could not read input Size.\n");
        exit(1);
      }

      if (fscanf(stdin, "%d", &iter_max_num) != 1) {
        printf("Error: could not read Iter_Max_Num Size.\n");
        exit(1);
      }
    }

    // broadcast do tamanho do grid e número máximo de iterações para cada processo 
    MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iter_max_num, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // allocate memory to the grid (matrix)
    grid = (double **) malloc(size * sizeof(double *));
    new_grid = (double **) malloc(size * sizeof(double *));

    for(int i = 0; i < size; i++){
        grid[i] = (double *) malloc(size * sizeof(double));
        new_grid[i] = (double *) malloc(size * sizeof(double));
    }

    //inicializar grid
    int center = size / 2;
    int linf = center - (size / 10);
    int lsup = center + (size / 10);
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            // inicializa região de calor no centro do grid
            if ( i >= linf && i <= lsup && j >= linf && j <= lsup)
                grid[i][j] = 100;
            else
                grid[i][j] = 0;
         }
     }
    

// número de linhas de cada processo
    int n_rows = size / num_procs;
    n_rows = floor(n_rows);
    int n_extra_rows = size % num_procs;

    if (rank < n_extra_rows) {
        n_rows += 1;
    }


    int start_row;
    if (rank < n_extra_rows) {
        start_row = rank * (n_rows + 1);
    } else {
        start_row = rank * n_rows + n_extra_rows;
    }

    // alocação
    double *grid_rows = (double *) malloc((n_rows+2) * size * sizeof(double));
    double *new_grid_rows = (double *) malloc((n_rows+2) * size * sizeof(double));
    double *aux;


    //cada processo copia de grid sua parte em grid_rows
    for(int i = 0 ; i < n_rows ; i++){
      for(int j = 0; j < size; j++){
        
          grid_rows[((i+1) * size) + j] = grid[i+start_row][j];     
        }
    }

    //Loop do cálculo do método iterativo de jacobi
    do{
        


        // Comunicação entre pares
    if (rank % 2 == 0) {
        if (rank != 0) {
            MPI_Sendrecv(&grid_rows[size], size, MPI_DOUBLE, rank-1, tag_top, &grid_rows[0], size, MPI_DOUBLE, rank-1, tag_bottom, MPI_COMM_WORLD, &status);
        }
        if (rank != num_procs-1) {
            MPI_Sendrecv(&grid_rows[n_rows*size], size, MPI_DOUBLE, rank+1, tag_bottom, &grid_rows[(n_rows+1)*size], size, MPI_DOUBLE, rank+1, tag_top, MPI_COMM_WORLD, &status);
        }

        
    }
    // Comunicação entre ímpares
    else {
        if (rank != num_procs-1) {
            MPI_Sendrecv(&grid_rows[n_rows*size], size, MPI_DOUBLE, rank+1, tag_bottom, &grid_rows[(n_rows+1)*size], size, MPI_DOUBLE, rank+1, tag_top, MPI_COMM_WORLD, &status);
        }
        if (rank != 0) {
            MPI_Sendrecv(&grid_rows[size], size, MPI_DOUBLE, rank-1, tag_top, &grid_rows[0], size, MPI_DOUBLE, rank-1, tag_bottom, MPI_COMM_WORLD, &status);
        }
    }


        hasError = 0;
       diffnorm = 0.0;
        
        //cálculo de cada processo
        if(rank==0){
          for(int i = 2; i <= n_rows; i++) {
            for(int j = 1; j < size-1; j++) {
                new_grid_rows[(i * size) + j] = 0.25 * (grid_rows[((i-1) * size) + j] + grid_rows[((i+1) * size) + j] +
                                                        grid_rows[(i * size) + j-1] + grid_rows[(i * size) + j+1]);

                // calculate error
                double diff = new_grid_rows[(i * size) + j] - grid_rows[(i * size) + j];
                diffnorm += diff * diff;
            }
        }}
        else if(rank==num_procs-1){
          
          for(int i = 1; i < n_rows; i++) {
            for(int j = 1; j < size-1; j++) {
                new_grid_rows[(i * size) + j] = 0.25 * (grid_rows[((i-1) * size) + j] + grid_rows[((i+1) * size) + j] +
                                                        grid_rows[(i * size) + j-1] + grid_rows[(i * size) + j+1]);

                // calculate error
                double diff = new_grid_rows[(i * size) + j] - grid_rows[(i * size) + j];
                diffnorm += diff * diff;
            }
        }}
        else{

          for(int i = 1; i <= n_rows; i++) {
            for(int j = 1; j < size-1; j++) {
                new_grid_rows[(i * size) + j] = 0.25 * (grid_rows[((i-1) * size) + j] + grid_rows[((i+1) * size) + j] +
                                                        grid_rows[(i * size) + j-1] + grid_rows[(i * size) + j+1]);

                // calculate error
                double diff = new_grid_rows[(i * size) + j] - grid_rows[(i * size) + j];
                diffnorm += diff * diff;
            }
        }}

        //reduce error with all processes
        MPI_Allreduce(&diffnorm, &gdiffnorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

         //troca de ponteiros
         aux = grid_rows;
         grid_rows = new_grid_rows;
         new_grid_rows = aux;

       // check convergence
         if (sqrt(gdiffnorm) > CONV_THRESHOLD) {
        hasError = 1;
         }

        iter++;

    }while(hasError && iter < iter_max_num);

    double *all_grid_rows = NULL;

    // alocação da matriz final
    if(rank == 0) {
        all_grid_rows = (double *) malloc(size * size * sizeof(double));
    }

    int *displs = (int*)malloc(num_procs * sizeof(int));
    int *rcounts = (int*)malloc(num_procs * sizeof(int));

    for (int i = 0; i < num_procs; i++) {

        //rcounts[i] indica quantos elementos são recebidos pelo processo i
        rcounts[i] =  (n_rows*size);

        //displs[i] indica a posição inicial no buffer de recepção para armazenar os dados enviados pelo processo i.
        displs[i] = (i == 0) ? 0 : (displs[i-1] + rcounts[i-1]);
    }

    // gather grid_rows from all processes to the process 0

    if(size%num_procs==0){
    MPI_Gather(&grid_rows[size], n_rows * size, MPI_DOUBLE, all_grid_rows, n_rows * size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    else {
    MPI_Gatherv(&grid_rows[size], n_rows*size, MPI_DOUBLE, all_grid_rows, rcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    //print do grid final
    if(rank==0){
          
       for(int i = 0; i < size; i++){
           for(int j = 0; j < size; j++){
                printf("%lf ", all_grid_rows[i * size + j]);
           }
           printf("\n");
        }}
    
    //desalocar 
    for(int i = 0; i < size; i++){
        free(grid[i]);
    }
    free(grid);
    free(new_grid_rows);
    free(grid_rows);

    MPI_Finalize();

    return 0;
}