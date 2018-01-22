#include "../include/CudaVersion2.cuh"

CudaVersion2::CudaVersion2(std::string graphFilename, unsigned vertexesNumber) {
	this->init(graphFilename, vertexesNumber);
	tab_sizes = new int[vertexesNumber];

	for (int i = 0; i < vertexesNumber; i++) {
		tab_sizes[i] = 0;
		for (int j = 0; j < vertexesNumber; j++)
			linear_matrix[i * vertexesNumber + j] = -linear_matrix[i * vertexesNumber + j];
	}

	countEdges();
}

void CudaVersion2::countEdges() {
	int i, j, size;
	for (int i = 0; i < vertexesNumber; i++) {
		for (int j = i; j < vertexesNumber; j++) {
			if (linear_matrix[i * vertexesNumber + j] != 0) {
				tab_sizes[i]++;
			}
		}
	}
}

AbstractGraph::path* CudaVersion2::getCriticalPath(unsigned vertexStart) {
	path* res = new path();

	cudaDeviceReset();

	std::pair<std::vector<int>, std::vector<unsigned>> pair;
	bf(vertexStart, &pair);
	int intIndex = std::min_element(pair.first.begin(), pair.first.end()) - pair.first.begin();
	res->pathLength = -pair.first[intIndex];

	return res;
}

AbstractGraph::path * CudaVersion2::getCriticalPath() {
	return getCriticalPath(0);
}

// ---------------------------------------------------------------------------------------------

__global__ void relax(unsigned vertexesNumber, unsigned edgesAmount, int edgeStart, int* cuda_matrix, int* cuda_distance) {
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int weight;

	//printf("id = %d, blockIdx = %d, threardIdx = %d\n", id, blockIdx.x, threadIdx.x);

	if (id >= edgesAmount) return;
	//printf("dupa2\n");


	weight = cuda_matrix[edgeStart * vertexesNumber + id];
	//printf("dupa3\n");
	if (weight != 0) {
		//printf("dupa4\n");

		if (cuda_distance[id] > cuda_distance[edgeStart] + weight) {
			//printf("dupa6\n");

			atomicMin(&(cuda_distance[id]), (cuda_distance[edgeStart] + weight));
			//printf("dupa7\n");

		}
	}

}

void CudaVersion2::bf(unsigned row, std::pair<std::vector<int>, std::vector<unsigned>>* pair) {
	int* distance = new int[vertexesNumber];
	int* return_distance = new int[vertexesNumber];

	int* cuda_distance;
	int* cuda_matrix;

	std::vector<unsigned> predecessor;

	cudaMalloc(&cuda_distance, sizeof(int) * vertexesNumber);
	cudaMalloc(&cuda_matrix, sizeof(int) * vertexesNumber * vertexesNumber);

	for (int i = 0; i < vertexesNumber; i++) {
		distance[i] = INT_MAX;
	}

	distance[row] = 0;

	cudaMemcpy(cuda_distance, distance, sizeof(int) * vertexesNumber, cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_matrix, linear_matrix, sizeof(int) * vertexesNumber * vertexesNumber, cudaMemcpyHostToDevice);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	int b, t, edgesAmount;

	for (int i = 0; i < vertexesNumber; i++) { // wywolujemy tyle watkow ile mamy par

		edgesAmount = tab_sizes[i];
		//std::cout << "jebac4: " << edgesAmount << std::endl;

		if (edgesAmount == 0) {
			continue;
		}

		b = (edgesAmount / 24) + 1; // liczba blokow
		if (b == 1)
			t = edgesAmount;
		else
			t = 24;
		//std::cout << "b: " << b << "t: " << t << std::endl;
		relax <<<b, t>>> (vertexesNumber, edgesAmount, i, cuda_matrix, cuda_distance);
	}

	cudaDeviceSynchronize();

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&miliseconds, start, stop);

	cudaMemcpy(return_distance, cuda_distance, sizeof(int) * vertexesNumber, cudaMemcpyDeviceToHost);

	cudaFree(cuda_distance);
	cudaFree(cuda_matrix);

	for (int k = 0; k < vertexesNumber; k++)
		std::cout << return_distance[k] << std::endl;

	pair->first = std::vector<int>(return_distance, return_distance + vertexesNumber);
	pair->second = predecessor;
}