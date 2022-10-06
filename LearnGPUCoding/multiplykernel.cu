#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

// define 
cudaError_t vectorMultiplyCuda(int* out_vec, const int* in_a, const int* in_b, unsigned int size); 

// write the kernel fuction 
__global__ void multiplyKernel(int* c, const int* a, const int* b, const unsigned int size) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < size)
		c[i] = a[i] * b[i]; 
}
bool checkForCudaError(cudaError_t err) {
	if (err != cudaSuccess) {
		fprintf(stderr, "cuda error");
		return false;
	}
	return true;
}

cudaError_t vectorMultiplyCuda(int* c, const int* a, const int* b, unsigned int size) {
	// declare variables. 
	int* dev_a = 0; 
	int* dev_b = 0; 
	int* dev_c = 0; 
	cudaError_t cudaStatus; 

	cudaStatus = cudaSetDevice(0); 
	if (!checkForCudaError(cudaStatus)) {
		goto Error; 
	}

	// Allocate GPU buffers for input (2) and output (1) vectors
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int)); 
	if (!checkForCudaError(cudaStatus)) {
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int)); 
	if (!checkForCudaError(cudaStatus)) {
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int)); 
	if (!checkForCudaError(cudaStatus)) {
		goto Error;
	}

	// copy input vectors from host to GPU buffers
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice); 
	if (!checkForCudaError(cudaStatus)) {
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
	if (!checkForCudaError(cudaStatus)) {
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_c, c, size * sizeof(int), cudaMemcpyHostToDevice);
	if (!checkForCudaError(cudaStatus)) {
		goto Error;
	}
	int threadsPerBlock = 256; 
	int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock; 

	multiplyKernel <<<blocksPerGrid, threadsPerBlock>>> (dev_c, dev_a, dev_b, size); 

	cudaStatus = cudaGetLastError(); 
	if (!checkForCudaError(cudaStatus)) {
		fprintf(stderr, "multiplyKernel launch failed: %s\n", cudaGetErrorString(cudaStatus)); 
		goto Error; 
	}

	cudaStatus = cudaDeviceSynchronize(); 
	if (!checkForCudaError(cudaStatus)) {
		fprintf(stderr, "Thread synchronization failed."); 
		goto Error;
	}

	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost); 
	if (!checkForCudaError(cudaStatus)) {
		fprintf(stderr, "failed to copy from host to device."); 
		goto Error; 
	}

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);
	return cudaStatus;
}


int main() {

	const unsigned int size = 5000; 
	int a[size]; 
	int b[size]; 
	int c[size]; 
	int i; 
	
	for (i = 0; i < size; i++) {
		a[i] = i; 
		b[i] = i; 
		c[i] = 0; 
	}
	
	cudaError_t status = vectorMultiplyCuda(c, a, b, size); 
	
	if (!checkForCudaError(status)) {
		fprintf(stderr, "multiply with cuda failed"); 
		return 1; 
	}
	status = cudaDeviceReset(); 
	int j; 
	for (j = 0; j < size - 1; j++) {
		printf("%d \n", c[j]); 
	}

	return 0; 
}
bool checkPtrForNull(int* ptr) {
	if (ptr == NULL) {
		fprintf(stderr, "failed to allocate pointer."); 
	}
	exit(EXIT_FAILURE); 
}

