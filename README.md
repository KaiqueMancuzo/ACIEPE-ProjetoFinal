# ACIEPE-ProjetoFinal

## Descrição do problema:

Jacobi Iterative Method that Solves the Laplace Equation

The code implements a Jacobi iterative method that solves the Laplace equation for heat transfer. The Jacobi iterative method is a means for iteratively calculating the solution to a differential equation by continuously refining the solution until the answer has converged upon a stable solution or some fixed number of steps have completed and the answer is either deemed good enough or unconverged. The example code represents a 2D plane of material that has been divided into a grid of equally sized cells. As heat is applied to the center of this plane, the Laplace equation dictates how the heat will transfer from grid point to grid point over time. To calculate the temperature of a given grid point for the next time iteration, one simply calculates the average of the temperatures of the neighboring grid points from the current iteration. Once the next value for each grid point is calculated, those values become the current temperature and the calculation continues. At each step the maximum temperature change across all grid points will determine if the problem has converged upon a steady state.

![image](https://user-images.githubusercontent.com/127627071/229187811-c593d43f-6789-4987-ba05-354ae6c707c2.png)

The Figure E.1 below shows the result of applying the algorithm on a surface represented by a 512 x 512 matrix with 10000 steps.

## Paralelização com OpenMP:

Este código resolve a equação de Laplace em uma grade 2D regular usando iteração de Jacobi simples. A estratégia de paralelização utilizada aqui é paralelizar o loop interno (j) com a diretiva OpenMP pragma omp parallel for.

```

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

```

A diretiva pragma omp parallel for instrui o compilador a executar o loop interno em paralelo, com cada thread executando uma parte da iteração. A opção collapse(2) indica que os dois loops (i e j) podem ser combinados em um loop, o que pode melhorar o desempenho da paralelização.

As variáveis grid, new_grid e hasError são compartilhadas entre as threads por meio da cláusula shared. A variável hasError é atualizada dentro do loop interno e usada posteriormente no loop principal para determinar se o algoritmo convergiu ou não. A atualização dos valores da matriz new_grid é feita paralelamente pelas threads.


## Paralelização com MPI:

A estratégia de paralelização utilizada no código é a divisão do grid em sub-grids, em que cada sub-grid é atribuído a um processo.

O número de linhas atribuídas a cada processo é calculado pela divisão inteira do tamanho do grid pelo número de processos. O processo 0 é responsável por ler o tamanho do grid e o número máximo de iterações a serem executadas a partir da entrada padrão e, em seguida, envia esses valores para todos os outros processos usando MPI_Bcast.

Durante a execução do método iterativo de Jacobi, cada processo realiza o cálculo para as linhas de seu sub-grid. Em seguida, os sub-grids são trocados com os vizinhos usando MPI_Send e MPI_Recv. Isso é feito em um loop até que o critério de convergência seja atingido ou o número máximo de iterações seja alcançado.

```

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

    iter++;

}while(hasError && iter < iter_max_num);

```
Finalmente, o processo 0 reúne todos os sub-grids em um único grid com MPI_GATHER.
```
//gather grid_rows from all processes to the process 0
     if(size%num_procs==0){
          MPI_Gather(&grid_rows[size], n_rows * size, MPI_DOUBLE, all_grid_rows,    n_rows * size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
          }
          else {
          MPI_Gatherv(&grid_rows[size], n_rows*size, MPI_DOUBLE, all_grid_rows, rcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      }
      
 ```
 
      
## Paralelização com CUDA:

A estratégia de paralelização utilizada neste código é a de dividir a matriz de entrada em blocos menores e processá-los paralelamente em blocos de threads. Para isso, foi utilizada a técnica de divisão em blocos (tiling), em que os dados são carregados em blocos menores e processados de forma independente, reduzindo assim o tempo de comunicação e aumentando a eficiência do código.

A técnica de divisão em blocos é utilizada no início da função Jacobi, com o cálculo dos índices bx e by que definem o bloco atual a ser processado, e dos índices tx e ty que definem a thread atual dentro do bloco. Em seguida, a matriz é carregada em blocos menores para a memória compartilhada da GPU, por meio da variável tile.

Após o carregamento dos dados na memória compartilhada, o código calcula a nova solução do sistema de equações, comparando os valores de phi e phi_old em cada iteração. Para cada bloco, a função calcula a diferença entre a solução atual e a solução anterior, armazenando o valor em diff.

Depois disso, o valor de diff é armazenado na memória compartilhada na variável s_max_diff. A função atomicMax_double é utilizada para atualizar o valor máximo de diferença max_diff, de forma que todos os blocos compartilhem o mesmo valor atualizado. Por fim, a função Jacobi é executada em um loop, com o número de iterações determinado pela variável MAX_ITER.


```

// Jacobi iteration
    while (hasError && iter < MAX_ITER) {
        hasError = 0;
        // CHANGE 2 - Now correctly doing the grid assignment to old grid
        d_temp = d_phi_old;
        d_phi_old = d_phi;
        d_phi = d_temp;


        // launch the kernel
        diff = 0;
        cudaMemcpy(d_diff, &diff, sizeof(double), cudaMemcpyHostToDevice);
        Jacobi<<<grid, block>>>(d_phi, d_phi_old, d_diff, N);

        // copy the diff value back to the host
        cudaMemcpy(&diff, d_diff, sizeof(double), cudaMemcpyDeviceToHost);

        // check for convergence
        if (diff > CONV_THRESHOLD) {
            hasError = 1;
        }

        iter++;
    }
    
    
 ```
    
    
## Análise da escalabilidade: esperada e obtida

## Discussão sobre a eficiência da solução;

## Conclusões;

