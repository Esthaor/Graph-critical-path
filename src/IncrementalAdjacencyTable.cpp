#include "../include/IncrementalAdjacencyTable.h"

IncrementalAdjacencyTable::IncrementalAdjacencyTable(std::string graphFilename, unsigned vertexesNumber) {
	this->init(graphFilename, vertexesNumber);
	adjacency_table = new edges[vertexesNumber];
	tab_sizes = new int[vertexesNumber];

	for (int i = 0; i < vertexesNumber; i++)	// ujemne wagi
		for (int j = 0; j < vertexesNumber; j++)
			matrix[i][j] = -matrix[i][j];

	fillAdjacencyTable();
}

void IncrementalAdjacencyTable::fillAdjacencyTable() {
	int i, j, size;
	for (i = 0; i < vertexesNumber; i++) {
		for (j = i; j < vertexesNumber; j++) {
			if (matrix[i][j] != 0) {
				adjacency_table[i].push_back(new std::pair<int, int>(j, matrix[i][j]));
			}
		}
	}

	tab = new std::pair<int, int>**[vertexesNumber];
	for (i = 0; i < vertexesNumber; i++) {
		tab_sizes[i]= adjacency_table[i].size();
		
		if (size == 0) {
			tab[i] = nullptr;// new std::pair<int, int>*[1];
			//tab[i][0] = new std::pair<int, int>(i, 0);
			continue;
		}
		else {
			tab[i] = new std::pair<int, int>*[tab_sizes[i]];
		}

		for (j = 0; j < tab_sizes[i]; j++) {
			tab[i][j] = new std::pair<int, int>(adjacency_table[i][j]->first, adjacency_table[i][j]->second);
		}
	}
}

AbstractGraph::path* IncrementalAdjacencyTable::getCriticalPath(unsigned vertexStart) {
	path* res = new path();

	std::pair<std::vector<int>, std::vector<unsigned>>* pair = bellmanFord(vertexStart);
	int intIndex = std::min_element(pair->first.begin(), pair->first.end()) - pair->first.begin();
	res->pathLength = -pair->first[intIndex];

	return res;
}

AbstractGraph::path* IncrementalAdjacencyTable::getCriticalPath() {
	return this->getCriticalPath(0);
}

std::pair<std::vector<int>, std::vector<unsigned>>* IncrementalAdjacencyTable::bellmanFord(unsigned row) {
	std::vector<int> distance;
	std::vector<unsigned> predecessor;

	for (int k = 0; k < vertexesNumber; k++) {
		distance.push_back(INT_MAX);
	}

	distance[row] = 0;
	edges vertexEdges;
	int endVertex, weight, size;

	for (int i = 0; i < vertexesNumber; i++) {
		//vertexEdges = adjacency_table[i];

		size = tab_sizes[i];

		for (int j = 0; j < size; j++) {
			if (tab[i] == nullptr) {
				continue;
			}
			endVertex = tab[i][j]->first;//vertexEdges.at(j)->first;
			weight = tab[i][j]->second;//vertexEdges.at(j)->second;
			if (weight != 0) {
				if (distance[endVertex] > distance[i] + weight) {
					distance[endVertex] = distance[i] + weight;
				}
			}
		}
	}

	/*int temp = INT_MAX;
	for (int k = 0; k < vertexesNumber; k++) {
		if (temp > distance[k]) {
			temp = distance[k];
			std::cout << distance[k] << std::endl;
		}
	}*/

	return new std::pair<std::vector<int>, std::vector<unsigned>>(distance, predecessor);
}

