#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

__global__
void vecAddKernel(float* A, float* B, float* C, int n)
{
    // Calculate global thread ID
    int id = blockIdx.x*blockDim.x + threadIdx.x;

    // Make sure we do not go out of bounds
    if(id < n)
    {
        C[id] = A[id] + B[id];
    }
}


void vecAdd(float* h_A, float* h_B, float* h_C, int n)
{
	int size = n * sizeof(float);
	float *d_A, *d_B, *d_C;
	
	// 1: Allocate device memory for A, B, and C    
    //    Copy A and B to device memory
    cudaMalloc((void **) &d_A, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMalloc((void **) &d_B, size);
    cudaMemcpy(d_B, h_A, size, cudaMemcpyHostToDevice);
    cudaMalloc((void **) &d_C, size);
	
    // 2: Kernal launch code
	vecAddKernel<<<ceil(n/256.0), 256>>>(d_A, d_B, d_C, n);


    // 3: copy C from the device memory
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}


int main()
{

    // Generate an array with 1000000 random integer numbers
    int n = 1000000;
    float* h_A = (float*)malloc(n*sizeof(float));
    float* h_B = (float*)malloc(n*sizeof(float));
    float* h_C = (float*)malloc(n*sizeof(float));
    for(int i = 0; i < n; i++)
    {
        h_A[i] = rand() % 100;
        h_B[i] = rand() % 100;
    }

    // Call the function that does the vector addition
    vecAdd(h_A, h_B, h_C, n);

    // Print the first 10 elements of the resulting vector
    for(int i = 0; i < 10; i++)
    {
        printf("%f ", h_C[i]);
    }
    printf("\n");

    // Free the memory allocated
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
