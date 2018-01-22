#include "../include/CudaVersion3.cuh"

CudaVersion3::CudaVersion3(std::string graphFilename, unsigned vertexesNumber) {
	this->init(graphFilename, vertexesNumber);
	adjacency_table = new edges[vertexesNumber];
	tab_sizes = new int[vertexesNumber];

	for (int i = 0; i < vertexesNumber; i++)	// ujemne wagi
		for (int j = 0; j < vertexesNumber; j++)
			matrix[i][j] = -matrix[i][j];

	fillAdjacencyTable();
}

void CudaVersion3::fillAdjacencyTable() {
	int i, j, size;
	for (i = 0; i < vertexesNumber; i++) {
		for (j = i; j < vertexesNumber; j++) {
			if (matrix[i][j] != 0) {
				adjacency_table[i].push_back(new std::pair<int, int>(j, matrix[i][j]));
			}
		}
	}

	tab = new std::pair<int, int>*[vertexesNumber];
	for (i = 0; i < vertexesNumber; i++) {
		tab_sizes[i] = adjacency_table[i].size();

		if (size == 0) {
			tab[i] = nullptr;
			continue;
		}

		tab[i] = new std::pair<int, int>[tab_sizes[i]];


		for (j = 0; j < tab_sizes[i]; j++) {
			tab[i][j] = std::pair<int, int>(adjacency_table[i][j]->first, adjacency_table[i][j]->second);
		}
	}
}

AbstractGraph::path* CudaVersion3::getCriticalPath(unsigned vertexStart) {
	path* res = new path();

	cudaDeviceReset();

	std::pair<std::vector<int>, std::vector<unsigned>> pair;
	bf(vertexStart, &pair);
	int intIndex = std::min_element(pair.first.begin(), pair.first.end()) - pair.first.begin();
	res->pathLength = -pair.first[intIndex];

	return res;
}

AbstractGraph::path * CudaVersion3::getCriticalPath() {
	return getCriticalPath(0);
}

__global__ void initNodeWeight_adj(unsigned row, unsigned vertexesNumber, int* cuda_distance) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id >= vertexesNumber) return;

	cuda_distance[id] = INT_MAX;
	if (id == row)
		cuda_distance[row] = 0;

}

__global__ void relax_adj(unsigned vertexesNumber, unsigned edgesAmount, int edgeStart, std::pair<int, int>** cuda_adjacency_table, int* cuda_distance) {
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id >= edgesAmount) return;

	if (cuda_adjacency_table[edgeStart] == nullptr) return;
	int endVertex = cuda_adjacency_table[edgeStart][id].first;
	int weight = cuda_adjacency_table[edgeStart][id].second;

	if (cuda_distance[endVertex] > cuda_distance[edgeStart] + weight) {
		atomicMin((cuda_distance + sizeof(int) * endVertex), (cuda_distance[edgeStart] + weight));
	}
}

void CudaVersion3::bf(unsigned row, std::pair<std::vector<int>, std::vector<unsigned>>* pair) {
	std::cout << "bf1" << std::endl;

	int* distance = new int[vertexesNumber];
	int* return_distance = new int[vertexesNumber];

	int* cuda_distance;
	//int* cuda_matrix;

	std::pair<int, int>** cuda_adjacency;
	std::pair<int, int>** cuda_adjacency_to_copy = new std::pair<int, int>*[vertexesNumber];

	std::pair<int, int>* temp_tab_pair;
	//std::pair<int, int>* temp_pair;

	std::vector<unsigned> predecessor;

	dim3 blocks(blocksNumber);
	dim3 threads(threadsNumber);

	cudaMalloc(&cuda_adjacency, sizeof(std::pair<int, int>*) * vertexesNumber);
	std::cout << "bf2" << std::endl;


	for (int m = 0; m < vertexesNumber; m++) {
		std::cout << "bf2-----------------------------------" << std::endl;

		cudaMalloc(&temp_tab_pair, sizeof(std::pair<int, int>) * tab_sizes[m]);

		for (int n = 0; n < tab_sizes[m]; n++) {
			cudaMemcpy(&temp_tab_pair[n], &tab[m][n], sizeof(std::pair<int, int>), cudaMemcpyHostToDevice);
		}

		std::cout << "bf2###############################################################################################" << std::endl;


		cuda_adjacency_to_copy[m] = temp_tab_pair;
	}
		std::cout << "bf3" << std::endl;


	cudaMalloc(&cuda_distance, sizeof(int) * vertexesNumber);

	cudaMemcpy(cuda_adjacency, cuda_adjacency_to_copy, sizeof(std::pair<int, int>*) * vertexesNumber, cudaMemcpyHostToDevice);
	std::cout << "bf4" << std::endl;

	for (int i = 0; i < vertexesNumber; i++) {
		distance[i] = INT_MAX;
	}

	distance[row] = 0;
	cudaMemcpy(cuda_distance, distance, sizeof(int) * vertexesNumber, cudaMemcpyHostToDevice);

	//initNodeWeight <<<blocksNumber, threadsNumber>>>(row, vertexesNumber, cuda_distance);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	int b, t, edgesAmount;
	std::cout << "bf5" << std::endl;

	for (int i = 0; i < vertexesNumber; i++) { // wywolujemy tyle watkow ile mamy par

		edgesAmount = tab_sizes[i];

		if (edgesAmount == 0) {
			continue;
		}
		b = (edgesAmount / 24) + 1; // liczba blokow
		if (b == 1)
			t = edgesAmount;
		else
			t = 24;
		std::cout << "bf6" << std::endl;

		relax_adj << <b, t >> > (vertexesNumber, edgesAmount, i, cuda_adjacency, cuda_distance);

	}

	cudaDeviceSynchronize();

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&miliseconds, start, stop);

	cudaMemcpy(return_distance, cuda_distance, sizeof(int) * vertexesNumber, cudaMemcpyDeviceToHost);

	for (int h = 0; h < vertexesNumber; h++) {
		cudaFree(cuda_adjacency[h]);
	}
	cudaFree(cuda_adjacency);
	cudaFree(cuda_distance);

	pair->first = std::vector<int>(return_distance, return_distance + vertexesNumber);
	pair->second = predecessor;
}