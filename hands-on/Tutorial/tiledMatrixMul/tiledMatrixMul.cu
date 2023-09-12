#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
// Time measurement
#include <sys/time.h>


#define TILE_WIDTH 16


__global__ 
void MatrixMulKernel(int m, int n, int k, float* A, float* B, float* C)
{
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int Row = by * blockDim.y + ty;
    int Col = bx * blockDim.x + tx;
    float Cvalue = 0;

    // Outer loop: how many tiles are there in A and B
    for (int t = 0; t < n / TILE_WIDTH; ++t) {

        // Loading A and B tiles into shared memory
        ds_A[ty][tx] = A[Row*n + t*TILE_WIDTH+tx];
        ds_B[ty][tx] = B[(t*TILE_WIDTH + ty) * k + Col];
        __syncthreads();

        // Computing in a tile
        for (int i = 0; i < TILE_WIDTH; ++i) {
            Cvalue += ds_A[ty][i] * ds_B[i][tx];
        }
        __syncthreads(); // Complete the work in the shared memory tile
    }
    C[Row*k + Col] = Cvalue;
}


// Generating a main function to test the above functions
int main(int argc, char** argv)
{
    int m = 1024;
    int n = 1024;
    int k = 1024;

    float* h_A = (float*)malloc(sizeof(float) * m * n);
    float* h_B = (float*)malloc(sizeof(float) * n * k);
    float* h_C = (float*)malloc(sizeof(float) * m * k);

    for (int i = 0; i < m * n; ++i) {
        h_A[i] = 1.0f;
    }
    for (int i = 0; i < n * k; ++i) {
        h_B[i] = 2.0f;
    }




    float* d_A; float* d_B; float* d_C;
    cudaMalloc((void**)&d_A, sizeof(float) * m * n);
    cudaMalloc((void**)&d_B, sizeof(float) * n * k);
    cudaMalloc((void**)&d_C, sizeof(float) * m * k);

    cudaMemcpy(d_A, h_A, sizeof(float) * m * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(float) * n * k, cudaMemcpyHostToDevice);

    dim3 dimGrid((k-1)/TILE_WIDTH+1, (m-1)/TILE_WIDTH+1, 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    // Print the information being multiplied and the hardware being used
    printf("Matrix A: %d x %d\n", m, n);
    printf("Matrix B: %d x %d\n", n, k);
    printf("Grid size: %d x %d\n", dimGrid.x, dimGrid.y);
    printf("Block size: %d x %d\n", dimBlock.x, dimBlock.y);


    // Measure the execution time
    struct timeval start, end;
    gettimeofday(&start, NULL);
    MatrixMulKernel<<<dimGrid, dimBlock>>>(m, n, k, d_A, d_B, d_C);

    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    printf("Elapsed time: %ld us\n", (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec));

    cudaMemcpy(h_C, d_C, sizeof(float) * m * k, cudaMemcpyDeviceToHost);

    // Verify the result
    for (int i = 0; i < m * k; ++i) {
        if (abs(h_C[i] - 2.0f * n) > 1e-5) {
            printf("Error!\n");
            break;
        }
    }
    printf("Success!\n");


    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}


