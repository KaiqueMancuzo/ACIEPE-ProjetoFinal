/*
    This program solves Laplace's equation on a regular 2D grid using simple Jacobi iteration.
    The stencil calculation stops when  iter > ITER_MAX
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>

#define CONV_THRESHOLD 1.0e-5f // threshold of convergence

// matrix to be solved
double **grid;

// auxiliary matrix
double **new_grid;

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

void print_grid(){

    for(int i = 0; i < size; i++){
        for(int j = 0; j < size; j++){
            printf("%lf ", grid[i][j]);
        }
        printf("\n");
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
    allocate_memory();

    initialize_grid();

    while ( hasError && iter < iter_max_num ) {
        hasError = 0;

        #pragma omp parallel for collapse(2) shared(grid, new_grid, hasError)
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
        
        double **aux = grid;
        grid = new_grid;
        new_grid = aux;

        iter++;
    }

    //print_grid();

    return 0;
}