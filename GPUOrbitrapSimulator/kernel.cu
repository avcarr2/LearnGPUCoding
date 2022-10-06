
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern const double eCharge = -1.60217663E-19;
extern const double daltonKg = 1.660539E-27;
extern int numberCharges = 1000000; 

__global__ void calcSine(double* c, const double* timeXfrequency, const double amplitude, const double frequency, unsigned int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < size)
		c[i] = sin(timeXfrequency[i]) * amplitude; 
}

bool checkForCudaError(cudaError_t err) {
	if (err != cudaSuccess) {
		fprintf(stderr, "cuda error");
		return false;
	}
	return true;
}
cudaError_t simulateTransient(double* c, const double* timeXfrequency, 
	const double amplitude, const double frequency, unsigned int size) {
	// declare variables. 
	double* dev_a = 0;
	double* dev_c = 0; 
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (!checkForCudaError(cudaStatus)) {
		goto Error;
	}

	// Allocate GPU buffers for input (2) and output (1) vectors
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(double));
	if (!checkForCudaError(cudaStatus)) {
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(double));
	if (!checkForCudaError(cudaStatus)) {
		goto Error;
	}

	// copy input vectors from host to GPU buffers
	cudaStatus = cudaMemcpy(dev_a, timeXfrequency, size * sizeof(double), cudaMemcpyHostToDevice);
	if (!checkForCudaError(cudaStatus)) {
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_c, c, size * sizeof(double), cudaMemcpyHostToDevice);
	if (!checkForCudaError(cudaStatus)) {
		goto Error;
	}
	int threadsPerBlock = 252;
	int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

	calcSine <<<blocksPerGrid, threadsPerBlock>>> (dev_c, dev_a, amplitude, frequency, size);

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

	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(double), cudaMemcpyDeviceToHost);
	if (!checkForCudaError(cudaStatus)) {
		fprintf(stderr, "failed to copy from host to device.");
		goto Error;
	}

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	return cudaStatus;
}

double daltonToMass(double dalton) {
    return dalton * daltonKg; 
}
double calculateAmplitude(double massKg, double chargeq, double numberIons, double frequency, 
    double lambdaR, double deltaZ) {
    return -chargeq * numberIons * frequency * deltaZ / lambdaR; 
}


int main()
{

    double startTime = 0; double endTime = 0.1;
    double samplingRateHz = 2E6;
    double timeStep = (endTime - startTime) / samplingRateHz;
    int numberTimeSteps = (int)(endTime / timeStep); 
	double* timeArray; 
	double* frequencyXtime;

    size_t timeArraySize = sizeof(double) * numberTimeSteps; 
    timeArray = (double*)malloc(timeArraySize); 
    if (timeArray == NULL) exit(EXIT_FAILURE); 

	frequencyXtime = (double*)malloc(timeArraySize);
	if (frequencyXtime == NULL) exit(EXIT_FAILURE);

    double frequency = 4.00E6; 
    int chargeState = 25; 
    int numberIons = numberCharges / chargeState; 
    double q = chargeState * eCharge; 

    double masskg = daltonToMass(2.5E4);
    double lambdaR = 0.9; 
    double amplitudeTerm = calculateAmplitude(masskg, q, numberIons, frequency, lambdaR, 0.1); 

	int i;
	for (i = 0; i < numberTimeSteps; i++) {
		frequencyXtime[i] = i * timeStep * frequency;
		timeArray[i] = 0; 
	}

	cudaError_t status = simulateTransient(timeArray, frequencyXtime, amplitudeTerm, frequency,numberTimeSteps); 

	if (!checkForCudaError(status)) {
		fprintf(stderr, "multiply with cuda failed");
		return 1;
	}
	status = cudaDeviceReset();
	
	return 0;
}
