/*
    This program solves Laplace's equation on a regular 2D grid using simple Jacobi iteration.

    The stencil calculation stops when  ( err >= CONV_THRESHOLD OR  iter > ITER_MAX )
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

#define CONV_THRESHOLD 1.0e-5f // threshold of convergence

// matrix to be solved
double **grid;

// auxiliary matrix
double **new_grid;

//criando uma variável auxiliar para tirar o loop de cópia
double **aux;


// size of each side of the grid
int size;
int iter_max_num;

// allocate memory for the grid
void allocate_memory(){
    grid = (double **) malloc(size * sizeof(double *));
    new_grid = (double **) malloc(size * sizeof(double *));

    for(int i = 0; i < size; i++){
        grid[i] = (double *) malloc(size * sizeof(double));
        new_grid[i] = (double *) malloc(size * sizeof(double));
    }
}

// initialize the grid
void initialize_grid(){
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
            new_grid[i][j] = 0.0;
        }
    }
}

// save the grid in a file
void save_grid(){

    for(int i = 0; i < size; i++){
        for(int j = 0; j < size; j++){
            fprintf(stdout, "%lf ", grid[i][j]);
        }
        fprintf(stdout, "\n");
    }
}

int main(int argc, char *argv[]){
    if (fscanf(stdin, "%d", &size) != 1) {
      printf("Error: could not read input Size.\n");
      exit(1);
    }

    if (fscanf(stdin, "%d", &iter_max_num) != 1) {
      printf("Error: could not read Iter_Max_Num Size.\n");
      exit(1);
    }
    
    int hasError = 1;
    int iter = 0;

    struct timeval time_start;
    struct timeval time_end;

    // allocate memory to the grid (matrix)
    allocate_memory();

    // set grid initial conditions
    initialize_grid();
    //save_grid();

    printf("Jacobi relaxation calculation: %d x %d grid\n", size, size);

    // get the start time
    gettimeofday(&time_start, NULL);

    // Jacobi iteration
    // This loop will end if either the maximum change reaches below a set threshold (convergence)
    // or a fixed number of maximum iterations have completed
    while ( hasError && iter < iter_max_num ) {

        hasError = 0;
        // calculates the Laplace equation to determine each cell's next value
        for( int i = 1; i < size-1; i++) {
            for(int j = 1; j < size-1; j++) {

                new_grid[i][j] = 0.25 * (grid[i][j+1] + grid[i][j-1] +
                                         grid[i-1][j] + grid[i+1][j]);

                double errorCurrent = fabs(new_grid[i][j] - grid[i][j]);
                if(errorCurrent > CONV_THRESHOLD ){
                    hasError = 1;                                        
                }     
            }
        }            
        
        aux = grid;
        grid = new_grid;
        
        // printf("\n\n----------\n");
        // save_grid();
         //printf("\n----------\n");
        
        new_grid = aux;

        iter++;
    }

    // get the end time
    gettimeofday(&time_end, NULL);

    double exec_time = (double) (time_end.tv_sec - time_start.tv_sec) +
                       (double) (time_end.tv_usec - time_start.tv_usec) / 1000000.0;

    //save the final grid in file
    save_grid();

    printf("\nKernel executed in %lf seconds with %d iterations.\n", exec_time, iter);

    return 0;
}