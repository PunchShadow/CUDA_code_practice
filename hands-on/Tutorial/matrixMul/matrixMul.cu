#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>

__global__
void MatrixMulKernel(int m, int n, int k, float* A, float* B, float* C)
{
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((Row < m) && (Col < k)) {
        float sum = 0;
        for (int i = 0; i < n; i++) {
            sum += A[Row * n + i] * B[i * k + Col];
        }
        C[Row * k + Col] = sum;
    }
}



__host__
void MatrixMulHost(int m, int n, int k)
{
    float* A = (float*)malloc(m * n * sizeof(float));
    float* B = (float*)malloc(n * k * sizeof(float));
    float* C = (float*)malloc(m * k * sizeof(float));

    for (int i = 0; i < m * n; i++)
    {
        A[i] = (float)rand() / RAND_MAX;
    }

    for (int i = 0; i < n * k; i++)
    {
        B[i] = (float)rand() / RAND_MAX;
    }

    // Allocate device memory
    float* d_A;
    float* d_B;
    float* d_C;

    // Add error checking
    cudaError_t err = cudaMalloc((void**)&d_A, m * n * sizeof(float));
    cudaError_t err1 = cudaMalloc((void**)&d_B, n * k * sizeof(float));
    cudaError_t err2 = cudaMalloc((void**)&d_C, m * k * sizeof(float));

    if (err != cudaSuccess || err1 != cudaSuccess || err2 != cudaSuccess)
    {
        printf("Error allocating device memory!\n");
        exit(1);
    }


    // Copy input matrices to device
    cudaMemcpy(d_A, A, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n * k * sizeof(float), cudaMemcpyHostToDevice);

    // Invoke kernel
    int TILE_WIDTH = 16;
    dim3 dimGrid((k - 1) / TILE_WIDTH + 1, (m - 1) / TILE_WIDTH + 1, 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    // Print the information being multiplied and the hardware being used
    printf("Matrix A: %d x %d\n", m, n);
    printf("Matrix B: %d x %d\n", n, k);
    printf("Grid size: %d x %d\n", dimGrid.x, dimGrid.y);
    printf("Block size: %d x %d\n", dimBlock.x, dimBlock.y);

    // Measure time
    struct timeval start, end;
    gettimeofday(&start, NULL);
    MatrixMulKernel<<<dimGrid, dimBlock>>>(m, n, k, d_A, d_B, d_C);
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    printf("Time: %ld us\n", (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec));
    
    // Copy result back to host
    cudaMemcpy(C, d_C, m * k * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify result
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < k; j++)
        {
            float sum = 0;
            for (int h = 0; h < n; h++)
            {
                sum += A[i * n + h] * B[h * k + j];
            }
            if (abs(C[i * k + j] - sum) > 0.01)
            {
                printf("C[%d][%d] = %f, should be %f\n", i, j, C[i * k + j], sum);
                printf("Wrong answer!\n");
                exit(1);
            }
        }
    }

    printf("Correct!\n");

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(A);
    free(B);
    free(C);
}

int main()
{
    int m = 1024;
    int n = 1024;
    int k = 1024;

    MatrixMulHost(m, n, k);

    return 0;
}