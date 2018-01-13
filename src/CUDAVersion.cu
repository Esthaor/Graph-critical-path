#include "../include/CUDAVersion.cuh"

CUDAVersion::CUDAVersion(std::string graphFilename, unsigned vertexesNumber){
	this->init(graphFilename, vertexesNumber);
}

AbstractGraph::path * CUDAVersion::getCriticalPath(unsigned vertexStart) {
	for (int i = 0; i < vertexesNumber; i++)	// ujemne wagi
		for (int j = 0; j < vertexesNumber; j++)
			matrix[i][j] = -matrix[i][j];

	path* res = new path();

	std::pair<std::vector<long>, std::vector<unsigned>> pair;
	bellmanFord(vertexStart, &pair);
	int intIndex = std::min_element(pair.first.begin(), pair.first.end()) - pair.first.begin();
	res->pathLength = -pair.first[intIndex];

	return res;
}

AbstractGraph::path * CUDAVersion::getCriticalPath() {
	return getCriticalPath(0);
}

__global__ void kernel(int vertexesNumber, long** matrix, long* distance) {
	int globalIndex = blockDim.x * blockIdx.x + threadIdx.x;
	int offset = blockDim.x * gridDim.x;

	if (globalIndex >= vertexesNumber)
		return;

	for (int i = 0; i < vertexesNumber; i++) {
		for (int j = globalIndex; j < vertexesNumber; j += offset) {
			if (matrix[i][j] != 0) {
				if (distance[j] > distance[i] + matrix[i][j]) {
					distance[j] = distance[i] + matrix[i][j];
				}
			}
		}
	}

}

void CUDAVersion::bellmanFord(unsigned row, std::pair<std::vector<long>, std::vector<unsigned>>* pair) {
	long* distance = new long[vertexesNumber];
	long* cuda_distance;
	long** cuda_matrix;
	std::vector<unsigned> predecessor;

	dim3 blocks(blocksNumber);
	dim3 threads(threadsNumber);

	cudaMalloc(&cuda_matrix, sizeof(long) * vertexesNumber * vertexesNumber);
	cudaMalloc(&cuda_distance, sizeof(long) * vertexesNumber);

	for (int i = 0; i < vertexesNumber; i++) {
		distance[i] = LONG_MAX;
	}

	distance[row] = 0;
	cudaMemcpy(cuda_distance, &distance, sizeof(long) * vertexesNumber, cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_matrix, matrix, sizeof(long) * vertexesNumber * vertexesNumber, cudaMemcpyHostToDevice);

	//void* args[] = { &vertexesNumber, cuda_matrix, cuda_distance};
	//cudaLaunchKernel((const void*)&kernel, blocks, threads, args);

	kernel<<<blocks, threads>>>(vertexesNumber, cuda_matrix, cuda_distance);
	cudaDeviceSynchronize();

	cudaMemcpy(&distance, cuda_distance, sizeof(long) * vertexesNumber, cudaMemcpyHostToDevice);

	cudaFree(cuda_matrix);
	cudaFree(cuda_distance);

	pair->first = std::vector<long>(distance, distance + vertexesNumber * sizeof(long));
	pair->second = predecessor;
}