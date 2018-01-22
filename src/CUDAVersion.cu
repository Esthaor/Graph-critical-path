#include "../include/CUDAVersion.cuh"

CUDAVersion::CUDAVersion(std::string graphFilename, unsigned vertexesNumber){
	this->init(graphFilename, vertexesNumber);
	tab_sizes = new int[vertexesNumber];

	for (int i = 0; i < vertexesNumber; i++)	// ujemne wagi
		for (int j = 0; j < vertexesNumber; j++)
			matrix[i][j] = -matrix[i][j];

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

	fillAdjacencyTable();
}

/*CUDAVersion::~CUDAVersion() {
	delete
}*/

void CUDAVersion::fillAdjacencyTable() {

	for (int i = 0; i < vertexesNumber; i++) {
		tab_sizes[i] = 0;
		for (int j = 0; j < vertexesNumber; j++) {
			if (matrix[i][j] <= 0) {
				stab[i * vertexesNumber + j].first = j;
				stab[i * vertexesNumber + j].second = matrix[i][j];
				tab_sizes[i]++;
				//tab_end[i].push_back(j);
			}
		}
	}
}

AbstractGraph::path* CUDAVersion::getCriticalPath(unsigned vertexStart) {
	path* res = new path();

	cudaDeviceReset();

	std::pair<std::vector<int>, std::vector<unsigned>> pair;
	bf(vertexStart, &pair);
	int intIndex = std::min_element(pair.first.begin(), pair.first.end()) - pair.first.begin();
	res->pathLength = -pair.first[intIndex];

	return res;
}

AbstractGraph::path * CUDAVersion::getCriticalPath() {
	return getCriticalPath(0);
}

__global__ void kernel_old(unsigned vertexesNumber, int* matrix, int* distance) {
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

__global__ void kernelNew_old(unsigned vertexesNumber, int* matrix, int* distance) {
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

void CUDAVersion::bellmanFord_old(unsigned row, std::pair<std::vector<int>, std::vector<unsigned>>* pair) {
	int* distance = new int[vertexesNumber];
	int* cuda_distance;
	int* cuda_matrix;
	std::vector<unsigned> predecessor;

	dim3 blocks(blocksNumber);
	dim3 threads(threadsNumber);

	cudaMalloc(&cuda_matrix, sizeof(int) * vertexesNumber * vertexesNumber);
	cudaMalloc(&cuda_distance, sizeof(int) * vertexesNumber);

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
	kernelNew_old<<<blocks, threads>>>(vertexesNumber, cuda_matrix, cuda_distance);
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

__global__ void relax_old(unsigned vertexesNumber, unsigned edgesAmount, int edgeStart, int* cuda_matrix, int* cuda_distance) {
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int sum, weight;

	printf("id = %d, blockIdx = %d, threardIdx = %d\n", id, blockIdx.x, threadIdx.x);

	if (id >= edgesAmount) return;
	printf("dupa2\n");


		weight = cuda_matrix[edgeStart * vertexesNumber + id];
			printf("dupa3\n");
		if (weight != 0) {
			printf("dupa4\n");

			sum = cuda_distance[edgeStart] + weight;
			printf("dupa5\n");

			if (cuda_distance[id] > weight) {
				printf("dupa6\n");

				atomicMin(&(cuda_distance[id]), sum);
				printf("dupa7\n");

			}
		}
	
}

void CUDAVersion::bf_old(unsigned row, std::pair<std::vector<int>, std::vector<unsigned>>* pair) {
	int* distance = new int[vertexesNumber];
	int* return_distance = new int[vertexesNumber];

	int* cuda_distance;
	int* cuda_matrix;

	std::vector<unsigned> predecessor;

	dim3 blocks(blocksNumber);
	dim3 threads(threadsNumber);

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

		if (edgesAmount == 0) continue;

		b = (edgesAmount / 24) + 1; // liczba blokow
		if (b == 1) t = edgesAmount; else t = 24;

		//std::cout << "b: " << b << "t: " << t << std::endl;
		relax_old <<<b, t>>> (vertexesNumber, edgesAmount, i, cuda_matrix, cuda_distance);

	}

	cudaDeviceSynchronize();

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&miliseconds, start, stop);

	cudaMemcpy(return_distance, cuda_distance, sizeof(int) * vertexesNumber, cudaMemcpyDeviceToHost);

	cudaFree(cuda_distance);
	cudaFree(cuda_matrix);

	//for (int k = 0; k < vertexesNumber; k++)
	//std::cout << return_distance[k] << std::endl;

	pair->first = std::vector<int>(return_distance, return_distance + vertexesNumber);
	pair->second = predecessor;
}

__global__ void initNodeWeight(unsigned row, unsigned vertexesNumber, int* cuda_distance) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id >= vertexesNumber) return;

	cuda_distance[id] = INT_MAX;
	if (id == row)
		cuda_distance[row] = 0;

}

__global__ void relax(unsigned edgesAmount, int edgeStart, std::pair<int, int>*** cuda_adjacency_table, int* cuda_distance) {
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	//if (threadIdx.x == 10 && blockIdx.x == 0)
	printf("id = %d, blockIdx = %d, threardIdx = %d\n", id, blockIdx.x, threadIdx.x);


	if (id >= edgesAmount) return;

	//if (threadIdx.x == 10)
		printf("dupa2\n");

	if (cuda_adjacency_table[edgeStart] == nullptr) return;
	//if (threadIdx.x == 10)
		printf("dupa3\n");

	int endVertex = cuda_adjacency_table[edgeStart][id]->first;
	//if (threadIdx.x == 10)
		printf("dupa4\n");

	int weight = cuda_adjacency_table[edgeStart][id]->second;

	//if (threadIdx.x == 10)
		printf("dupa5\n");

	if (cuda_distance[endVertex] > cuda_distance[edgeStart] + weight) {
		//if (threadIdx.x == 10)
			printf("dupa6\n");

		atomicMin((cuda_distance + sizeof(int) * endVertex), (cuda_distance[edgeStart] + weight));
	}
}

__global__ void greg(unsigned vertexesNumber, unsigned edgesAmount, int edgeStart, std::pair<int, int>* cuda_stab, int* cuda_distance) {
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id >= edgesAmount) return;

	int weight = cuda_stab[edgeStart * vertexesNumber + id].second;
	if (weight >= 0) return;

	int endVertex = cuda_stab[edgeStart * vertexesNumber + id].first;

	if (cuda_distance[endVertex] > cuda_distance[edgeStart] + weight) {
		atomicMin(&(cuda_distance[endVertex]), (cuda_distance[edgeStart] + weight));
	}
}


void CUDAVersion::bf(unsigned row, std::pair<std::vector<int>, std::vector<unsigned>>* pair) {
	int* distance = new int[vertexesNumber];
	int* return_distance = new int[vertexesNumber];

	int* cuda_distance;
	//int* cuda_matrix;

	std::pair<int, int>* cuda_stab;
	std::vector<unsigned> predecessor;

	dim3 blocks(blocksNumber);
	dim3 threads(threadsNumber);

	cudaMalloc(&cuda_stab, sizeof(std::pair<int,int>) * vertexesNumber * vertexesNumber);
	cudaMalloc(&cuda_distance, sizeof(int) * vertexesNumber);
	//cudaMalloc(&cuda_matrix, sizeof(int) * vertexesNumber * vertexesNumber);


	cudaMemcpy(cuda_stab, stab, sizeof(std::pair<int, int>) * vertexesNumber * vertexesNumber, cudaMemcpyHostToDevice);
	//cudaMemcpy(cuda_matrix, linear_matrix, sizeof(int) * vertexesNumber * vertexesNumber, cudaMemcpyHostToDevice);

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

	for (int i = 0; i < vertexesNumber; i++) { // wywolujemy tyle watkow ile mamy par
		
		edgesAmount = tab_sizes[i];
		//std::cout << "jebac4: " << edgesAmount << std::endl;

		if (edgesAmount == 0) {
			continue;
		}

/*		if (edgesAmount != tab_end[i].size()) {
			std::cout << "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" << std::endl;
		}*/
		
		b = (edgesAmount / 24) + 1; // liczba blokow
		if (b == 1)
			t = edgesAmount;
		else 
			t = 24;

		//if( i > 1950)
		//std::cout << "edgesAmount:\t" << edgesAmount << "\tb:\t" << b << "\tt:\t" << t << std::endl;
		greg <<<b, t>>> (vertexesNumber, edgesAmount, i, cuda_stab, cuda_distance);
		//relax_old <<<b, t>>> (vertexesNumber, edgesAmount, i, cuda_matrix, cuda_distance);

	}

	cudaDeviceSynchronize();

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&miliseconds, start, stop);

	cudaMemcpy(return_distance, cuda_distance, sizeof(int) * vertexesNumber, cudaMemcpyDeviceToHost);

	cudaFree(cuda_stab);
	cudaFree(cuda_distance);
	//cudaFree(cuda_matrix);

	/*int temp = INT_MAX;
	for (int k = 0; k < vertexesNumber; k++) {
		if (temp > return_distance[k]) {
			temp = return_distance[k];
			std::cout << return_distance[k] << std::endl;
		}
	}*/

	pair->first = std::vector<int>(return_distance, return_distance + vertexesNumber);
	pair->second = predecessor;
}