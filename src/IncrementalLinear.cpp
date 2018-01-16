#include"../include/IncrementalLinear.h"

IncrementalLinear::IncrementalLinear(std::string graphFilename, unsigned vertexesNumber) {
	this->init(graphFilename, vertexesNumber);
}

AbstractGraph::path* IncrementalLinear::getCriticalPath(unsigned vertexStart) {
	for (int i = 0; i < vertexesNumber; i++)	// ujemne wagi
		for (int j = 0; j < vertexesNumber; j++)
			linear_matrix[i * vertexesNumber + j] = -linear_matrix[i * vertexesNumber + j];

	path* res = new path();

	std::pair<std::vector<long>, std::vector<unsigned>>* pair = bellmanFord(vertexStart);
	int intIndex = std::min_element(pair->first.begin(), pair->first.end()) - pair->first.begin();
	res->pathLength = -pair->first[intIndex];

	return res;
}

// deprecated
std::vector<unsigned> IncrementalLinear::findPath(int index, std::vector<unsigned> predecessors) {
	int ite = index;
	std::vector<unsigned> vertexes;

	while (ite != 1) {
		vertexes.push_back(ite);
		ite = predecessors[ite - 1];
	}

	vertexes.push_back(ite);

	std::reverse(vertexes.begin(), vertexes.end());

	return vertexes;
}

AbstractGraph::path* IncrementalLinear::getCriticalPath() {
	return this->getCriticalPath(0);
}

std::pair<std::vector<long>, std::vector<unsigned>>* IncrementalLinear::bellmanFord(unsigned row) {
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
