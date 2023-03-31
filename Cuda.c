#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CONV_THRESHOLD 1.0e-4f // threshold of convergence
#define TILE_SIZE 16 

__device__ double atomicMax_double(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(fmax(val, __longlong_as_double(assumed))));
    } while (assumed != old);

    return __longlong_as_double(old);
}

__global__ void Jacobi(double* phi, double* phi_old, double* max_diff, int n) {
    __shared__ double tile[TILE_SIZE+2][TILE_SIZE+2];
    __shared__ double s_max_diff[TILE_SIZE*TILE_SIZE];
    
    double diff = 0.0;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int bx = blockIdx.x * blockDim.x;
    int by = blockIdx.y * blockDim.y;

    int i = bx + tx;
    int j = by + ty;

    // load the tile into shared memory
    if (i < n && j < n) {
        tile[tx+1][ty+1] = phi_old[i*n+j];
    }
    if (tx == 0 && i > 0) {
        tile[tx][ty+1] = phi_old[(i-1)*n+j];
    }
    if (tx == TILE_SIZE-1 && i < n-1) {
        tile[tx+2][ty+1] = phi_old[(i+1)*n+j];
    }
    if (ty == 0 && j > 0) {
        tile[tx+1][ty] = phi_old[i*n+(j-1)];
    }
    if (ty == TILE_SIZE-1 && j < n-1) {
        tile[tx+1][ty+2] = phi_old[i*n+(j+1)];
    }
    __syncthreads();

    if (i > 0 && i < n-1 && j > 0 && j < n-1) {
        double tmp = 0.25 * (tile[tx][ty+1] + tile[tx+2][ty+1] + tile[tx+1][ty] + tile[tx+1][ty+2]);
        diff = fabs(tmp - tile[tx+1][ty+1]);
        tile[tx+1][ty+1] = tmp;
    }
    
    __syncthreads();

    if (i < n && j < n) {
        phi[i*n + j] = tile[tx+1][ty+1];
    }

    // Change 3 - Put each diff value 
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    s_max_diff[tid] = diff;
    __syncthreads();

    for (int s=blockDim.x*blockDim.y/2; s>0; s/=2) {
        if (tid < s) {
            s_max_diff[tid] = fmax(s_max_diff[tid], s_max_diff[tid+s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMax_double(max_diff, s_max_diff[0]);
    }
}

int main(int argc, char *argv[]) {
    int N, MAX_ITER;

    fscanf(stdin, "%d", &N);
    fscanf(stdin, "%d", &MAX_ITER);

    double *phi, *phi_old, diff;
    double *d_phi, *d_phi_old, *d_diff, *d_temp;
    int i, j, iter = 0;
    int hasError = 1;
    size_t size = N * N * sizeof(double);

    // allocate memory on the host
    phi = (double*) malloc(size);
    phi_old = (double*) malloc(size);

    // initialize the grid
    int center = N / 2;
    int linf = center - (N / 10);
    int lsup = center + (N / 10);
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            // inicializa região de calor no centro do grid
            if ( i >= linf && i <= lsup && j >= linf && j <= lsup)
                phi[i*N + j] = 100;
            else
                phi[i*N + j] = 0;
            phi_old[i*N + j] = 0.0;
        }
    }

    // allocate memory on the device
    cudaMalloc(&d_phi, size);
    cudaMalloc(&d_phi_old, size);
    cudaMalloc(&d_diff, sizeof(double));

    // copy memory to the device
    cudaMemcpy(d_phi, phi, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_phi_old, phi_old, size, cudaMemcpyHostToDevice);

    // set the grid and block sizes
    dim3 grid((N+TILE_SIZE-1)/TILE_SIZE, (N+TILE_SIZE-1)/TILE_SIZE, 1);
    dim3 block(TILE_SIZE, TILE_SIZE, 1);

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

    // copy the solution back to the host
    cudaMemcpy(phi, d_phi, size, cudaMemcpyDeviceToHost);

    printf("\n\nIterations = %d\n\n", iter);
    printf("----FINAL MATRIX----\n");
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            //printf("%lf ", phi[i*N+j]);
        }
        //printf("\n");
    }

    return 0;