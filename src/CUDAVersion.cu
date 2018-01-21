#include "../include/CUDAVersion.cuh"

CUDAVersion::CUDAVersion(std::string graphFilename, unsigned vertexesNumber){
	this->init(graphFilename, vertexesNumber);

	if (true) {
		int device;
		cudaGetDevice(&device);
		cudaDeviceProp properties;
		cudaGetDeviceProperties(&properties, device);
		threadsNumber = properties.maxThreadsDim[0];
		blocksNumber = std::min(properties.maxGridSize[0], ((int)vertexesNumber + threadsNumber - 1) / threadsNumber);
	} else {
		blocksNumber = 1;
		threadsNumber = 32;
	}
}

AbstractGraph::path* CUDAVersion::getCriticalPath(unsigned vertexStart) {
	for (int i = 0; i < vertexesNumber; i++)	// ujemne wagi
		for (int j = 0; j < vertexesNumber; j++)
			linear_matrix[i * vertexesNumber + j] = -linear_matrix[i * vertexesNumber + j];

	path* res = new path();

	cudaDeviceReset();

	std::pair<std::vector<int>, std::vector<unsigned>> pair;
	bellmanFord(vertexStart, &pair);
	int intIndex = std::min_element(pair.first.begin(), pair.first.end()) - pair.first.begin();
	res->pathLength = -pair.first[intIndex];

	return res;
}

AbstractGraph::path * CUDAVersion::getCriticalPath() {
	return getCriticalPath(0);
}

__global__ void kernel(unsigned vertexesNumber, int* matrix, int* distance) {
	int globalIndex = blockDim.x * blockIdx.x + threadIdx.x;
	int offset = blockDim.x * gridDim.x;

	if (globalIndex >= vertexesNumber)
		return;

	for (int i = 0; i < vertexesNumber; i++) {
		for (int j = globalIndex; j < vertexesNumber; j += offset) {
			if (matrix[i * vertexesNumber + j] != 0) {
				if (distance[j] > distance[i] + matrix[i * vertexesNumber + j]) {
					distance[j] = distance[i] + matrix[i * vertexesNumber + j];
				}
			}
		}
	}
}

__global__ void kernelNew(unsigned vertexesNumber, int* matrix, int* distance) {
	int globalIndex = blockDim.x * blockIdx.x + threadIdx.x;
	int sum, weight;

	if (globalIndex >= vertexesNumber)
		return;
	
	for (int j = 0; j < vertexesNumber; j++) {
		weight = matrix[globalIndex * vertexesNumber + j];
		if (weight != 0) {
			sum = distance[globalIndex] + weight;
			if (distance[j] > weight) {
				atomicMin(&(distance[j]), sum);
			}
		}
	}
}

// aktualizacja distance po przejsciu edges
// wiêcej niz jedna wspolrzedna na kernel

void CUDAVersion::bellmanFord(unsigned row, std::pair<std::vector<int>, std::vector<unsigned>>* pair) {
	int* distance = new int[vertexesNumber];
	int* cuda_distance;
	int* cuda_matrix;
	std::vector<unsigned> predecessor;

	dim3 blocks(blocksNumber);
	dim3 threads(threadsNumber);

	cudaMalloc(&cuda_matrix, sizeof(int) * vertexesNumber * vertexesNumber);
	cudaMalloc(&cuda_distance, sizeof(long) * vertexesNumber);

	for (int i = 0; i < vertexesNumber; i++) {
		distance[i] = INT_MAX;
	}

	distance[row] = 0;
	cudaMemcpy(cuda_distance, distance, sizeof(int) * vertexesNumber, cudaMemcpyHostToDevice);
	//cudaMemcpy(cuda_distance, distance, sizeof(long) * vertexesNumber, cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_matrix, linear_matrix, sizeof(int) * vertexesNumber * vertexesNumber, cudaMemcpyHostToDevice);

	//void* args[] = { &vertexesNumber, cuda_matrix, cuda_distance};
	//cudaLaunchKernel((const void*)&kernel, blocks, threads, args);
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	//kernel<<<blocks, threads>>>(vertexesNumber, cuda_matrix, cuda_distance);
	kernelNew<<<blocks, threads>>>(vertexesNumber, cuda_matrix, cuda_distance);
	//kernelNew << <1, vertexesNumber>> >(vertexesNumber, cuda_matrix, cuda_distance);

	cudaDeviceSynchronize();

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&miliseconds, start, stop);

	cudaMemcpy(distance, cuda_distance, sizeof(int) * vertexesNumber, cudaMemcpyDeviceToHost);

	cudaFree(cuda_matrix);
	cudaFree(cuda_distance);

	//for (int k = 0; k < vertexesNumber; k++)
	//	std::cout << distance[k] << std::endl;

	pair->first = std::vector<int>(distance, distance + vertexesNumber);

	//for(int w = 0; w < vertexesNumber; w++)
	//	std::cout << pair->first[w] << std::endl;

	pair->second = predecessor;
}

// ---------------------------------------------------------------------------------------------

__global__ void initNodeWeight(unsigned row, unsigned vertexesNumber, int* cuda_matrix, int* cuda_distance) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id >= vertexesNumber) return;

	cuda_distance[id] = INT_MAX;
	if (id == row)
		cuda_distance[row] = 0;

}

__global__ void relax(unsigned vertexesNumber, int* matrix, int* distance) {
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int temp_index, sum;

	if (id >= vertexesNumber) return;

	for (int j = 0; j < vertexesNumber; j++) {
		temp_index = id * vertexesNumber + j;
		if (matrix[temp_index] != 0) {
			if (distance[j] >(sum = distance[id] + matrix[temp_index])) {
				atomicMin(&(distance[j]), sum);
			}
		}
	}
}

void CUDAVersion::bf(unsigned row, std::pair<std::vector<int>, std::vector<unsigned>>* pair) {
	int* distance = new int[vertexesNumber];
	int* cuda_distance;
	int* cuda_matrix;
	std::vector<unsigned> predecessor;

	dim3 blocks(blocksNumber);
	dim3 threads(threadsNumber);

	cudaMalloc(&cuda_matrix, sizeof(int) * vertexesNumber * vertexesNumber);
	cudaMalloc(&cuda_distance, sizeof(int) * vertexesNumber);

	cudaMemcpy(cuda_matrix, linear_matrix, sizeof(int) * vertexesNumber * vertexesNumber, cudaMemcpyHostToDevice);

	initNodeWeight << <blocksNumber, threadsNumber >> >(row, vertexesNumber, cuda_matrix, cuda_distance);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	relax <<<blocksNumber, threadsNumber>>>(vertexesNumber, cuda_matrix, cuda_distance);

	cudaDeviceSynchronize();

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&miliseconds, start, stop);

	cudaMemcpy(distance, cuda_distance, sizeof(int) * vertexesNumber, cudaMemcpyDeviceToHost);

	cudaFree(cuda_matrix);
	cudaFree(cuda_distance);

	pair->first = std::vector<int>(distance, distance + vertexesNumber);
	pair->second = predecessor;
}