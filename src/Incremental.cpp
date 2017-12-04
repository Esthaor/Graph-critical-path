#include"../include/Incremental.h"

Incremental::Incremental(std::string graphFilename, unsigned vertexesNumber) {
	this->init(graphFilename, vertexesNumber);
}

AbstractGraph::path* Incremental::getCriticalPath() {
	for (int i = 0; i < vertexesNumber; i++)
		for (int j = 0; j < vertexesNumber; j++)
			matrix[i][j] = -matrix[i][j];

	path* res = new path();

	int min = 0;
	int minStart = 0;
	int minEnd = 0;

	for (int i = 0; i < vertexesNumber; i++) {
		std::vector<long> longest = bellmanFord(i);
		int intIndex = std::min_element(longest.begin(), longest.end()) - longest.begin();
		int pathLength = longest[intIndex];

		if (pathLength < min) {
			min = pathLength;
			minStart = i;
			minEnd = intIndex;
		}
	}

	res->pathLength = -min;
	res->pathStart = minStart;
	res->pathEnd = minEnd;
	return res;
}

std::vector<long> Incremental::bellmanFord(unsigned row) {
	std::vector<long> distance;
	std::vector<unsigned> predecessor;

	for (int i = 0; i < vertexesNumber; i++) {
	//	for (int j = 0; j < vertexesNumber; j++) {
		//	if (matrix[i][j] != 0) {
				distance.push_back(LONG_MAX);
		//	}
	//	}
	}

	distance[row] = 0;
	
	for (int i = 0; i < vertexesNumber; i++) {
		for (int j = 0; j < vertexesNumber; j++) {
			if (matrix[i][j] != 0) {
				if (distance[j] > distance[i] + matrix[i][j]) {
					distance[j] = distance[i] + matrix[i][j];
					predecessor.push_back(j);
				}
			}
		}
	}
	
	return distance;
}