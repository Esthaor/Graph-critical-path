#include"../include/MpiVersion.h"

MpiVersion::MpiVersion(std::string graphFilename, unsigned vertexesNumber) {
	this->init(graphFilename, vertexesNumber);
}

AbstractGraph::path* MpiVersion::getCriticalPath(unsigned vertexStart) {
	for (int i = 0; i < vertexesNumber; i++)	// ujemne wagi
		for (int j = 0; j < vertexesNumber; j++)
			linear_matrix[i * vertexesNumber + j] = -linear_matrix[i * vertexesNumber + j];

	path* res = new path();

	std::pair<std::vector<long>, std::vector<unsigned>>* pair = bellmanFord(vertexStart);
	int intIndex = std::min_element(pair->first.begin(), pair->first.end()) - pair->first.begin();
	res->pathLength = -pair->first[intIndex];

	return res;
}

AbstractGraph::path* MpiVersion::getCriticalPath() {
	return this->getCriticalPath(0);
}

std::pair<std::vector<long>, std::vector<unsigned>>* MpiVersion::bellmanFord(unsigned row) {
	std::vector<long> distance;
	std::vector<unsigned> predecessor;

	for (int i = 0; i < vertexesNumber; i++) {
		distance.push_back(LONG_MAX);
	}

	distance[row] = 0;

	for (int i = 0; i < vertexesNumber; i++) {
		for (int j = 0; j < vertexesNumber; j++) {
			if (linear_matrix[i * vertexesNumber + j] != 0) {
				if (distance[j] > distance[i] + linear_matrix[i * vertexesNumber + j]) {
					distance[j] = distance[i] + linear_matrix[i * vertexesNumber + j];
					predecessor.push_back(j);
				}
			}
		}
	}

	return new std::pair<std::vector<long>, std::vector<unsigned>>(distance, predecessor);
}
