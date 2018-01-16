#include"../include/Incremental.h"

Incremental::Incremental(std::string graphFilename, unsigned vertexesNumber) {
	this->init(graphFilename, vertexesNumber);
}

AbstractGraph::path* Incremental::getCriticalPath(unsigned vertexStart) {
	for (int i = 0; i < vertexesNumber; i++)	// ujemne wagi
		for (int j = 0; j < vertexesNumber; j++)
			matrix[i][j] = -matrix[i][j];

	path* res = new path();

	std::pair<std::vector<long>, std::vector<unsigned>>* pair = bellmanFord(vertexStart);
	int intIndex = std::min_element(pair->first.begin(), pair->first.end()) - pair->first.begin();
	res->pathLength = -pair->first[intIndex];

//	res->vertexes = findPath(intIndex, pair->second);

	return res;
}

// deprecated
std::vector<unsigned> Incremental::findPath(int index, std::vector<unsigned> predecessors) {
	int ite = index;
	std::vector<unsigned> vertexes;

	while (ite != 1){
		vertexes.push_back(ite);	
		ite = predecessors[ite - 1];
	}

	vertexes.push_back(ite);
	
	std::reverse(vertexes.begin(), vertexes.end());

	return vertexes;
}

AbstractGraph::path* Incremental::getCriticalPath() {
	return this->getCriticalPath(0);
}

std::pair<std::vector<long>, std::vector<unsigned>>* Incremental::bellmanFord(unsigned row) {
	std::vector<long> distance;
	std::vector<unsigned> predecessor;

	for (int i = 0; i < vertexesNumber; i++) {
		distance.push_back(LONG_MAX);
	}

	distance[row] = 0;
	
	for (int i = 0; i < vertexesNumber; i++) {
		for (int j = 0; j < vertexesNumber; j++) { // odpowiada za wszystkie krawêdzie wychodz¹ce z wierzcholka i
			if (matrix[i][j] != 0) {				// pomijamy nieistniejace krawedzie
				if (distance[j] > distance[i] + matrix[i][j]) {		//jezeli dotychczasowy dystans jest wiekszy 
					distance[j] = distance[i] + matrix[i][j];
					predecessor.push_back(j);
				}
			}
		}
	}
	
	return new std::pair<std::vector<long>, std::vector<unsigned>>(distance, predecessor);
}