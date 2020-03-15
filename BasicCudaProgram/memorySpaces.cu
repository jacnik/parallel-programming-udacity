/*
    Using different memory spaces in CUDA

    Example code showing use of local, shared ang gobal memory.

    Rule to write fast code is to move frequently accessed data to fast memory.

    speed accesssing each memory types:
    local < shared << global << cpu ('host')

    Compile: nvcc memorySpaces.cu -o memorySpaces.out
    Run: ./memorySpaces
*/

#include <stdio.h>

#define NUM_BLOCKS 1
#define BLOCK_WIDTH 128


__global__ void shiftLeft(int* array)
{
    int idx = threadIdx.x;

    array[idx] = idx;

    __syncthreads();

    if (idx < BLOCK_WIDTH - 1) {
        int tmp = array[idx + 1];
        __syncthreads();
        array[idx] = tmp;
    }
}

int main(int argc,char **argv)
{
	const int ARRAY_SIZE = BLOCK_WIDTH;
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(int);

    // output array on the host
    int h_out[ARRAY_SIZE];

    // declare GPU memory pointer
    int* d_out;

	// allocate GPU memory
	cudaMalloc((void**) &d_out, ARRAY_BYTES);

    // launch the kernel
    shiftLeft<<<NUM_BLOCKS, BLOCK_WIDTH>>>(d_out);

	// copy back the result array to the CPU
	cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    // print out the resulting array
	for (int i = 0; i < ARRAY_SIZE; ++i) {
		printf("%d", h_out[i]);
		printf(i % 4 != 3 ? "\t" : "\n");
	}

    // free GPU memory allocation
	cudaFree(d_out);

    return 0;
}