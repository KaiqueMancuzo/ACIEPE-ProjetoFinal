# ACIEPE-ProjetoFinal

## Descrição do problema:

Jacobi Iterative Method that Solves the Laplace Equation

The code implements a Jacobi iterative method that solves the Laplace equation for heat transfer. The Jacobi iterative method is a means for iteratively calculating the solution to a differential equation by continuously refining the solution until the answer has converged upon a stable solution or some fixed number of steps have completed and the answer is either deemed good enough or unconverged. The example code represents a 2D plane of material that has been divided into a grid of equally sized cells. As heat is applied to the center of this plane, the Laplace equation dictates how the heat will transfer from grid point to grid point over time. To calculate the temperature of a given grid point for the next time iteration, one simply calculates the average of the temperatures of the neighboring grid points from the current iteration. Once the next value for each grid point is calculated, those values become the current temperature and the calculation continues. At each step the maximum temperature change across all grid points will determine if the problem has converged upon a steady state.

![image](https://user-images.githubusercontent.com/127627071/229187811-c593d43f-6789-4987-ba05-354ae6c707c2.png)

The Figure E.1 below shows the result of applying the algorithm on a surface represented by a 512 x 512 matrix with 10000 steps.

## Paralelização com OpenMP:

Este código resolve a equação de Laplace em uma grade 2D regular usando iteração de Jacobi simples. A estratégia de paralelização utilizada aqui é paralelizar o loop interno (j) com a diretiva OpenMP pragma omp parallel for.

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

A diretiva pragma omp parallel for instrui o compilador a executar o loop interno em paralelo, com cada thread executando uma parte da iteração. A opção collapse(2) indica que os dois loops (i e j) podem ser combinados em um loop, o que pode melhorar o desempenho da paralelização.

As variáveis grid, new_grid e hasError são compartilhadas entre as threads por meio da cláusula shared. A variável hasError é atualizada dentro do loop interno e usada posteriormente no loop principal para determinar se o algoritmo convergiu ou não. A atualização dos valores da matriz new_grid é feita paralelamente pelas threads.
