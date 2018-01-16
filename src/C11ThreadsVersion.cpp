#include"../include/C11ThreadsVersion.h"

C11ThreadsVersion::C11ThreadsVersion(std::string graphFilename, unsigned vertexesNumber) {
	this->init(graphFilename, vertexesNumber);
	distance = new long[vertexesNumber];
}

AbstractGraph::path * C11ThreadsVersion::getCriticalPath(unsigned vertexStart)
{
	for (int i = 0; i < vertexesNumber; i++)
		for (int j = 0; j < vertexesNumber; j++)
			matrix[i][j] = -matrix[i][j];

	for (int i = 0; i < vertexesNumber; i++) {
		distance[i] = LONG_MAX;
	}
	distance[vertexStart] = 0;

	path* res = new path();

	std::thread* threads = new std::thread[vertexesNumber];
	for (int i = 0; i < vertexesNumber; i++) {
		threads[i] = std::thread(&C11ThreadsVersion::bellmanFord, this, i, 0);
		threads[i].join();
	}

	std::vector<long> temp_distance = std::vector<long>(distance, distance + vertexesNumber);

	int intIndex = std::min_element(temp_distance.begin(), temp_distance.end()) - temp_distance.begin();
	res->pathLength = -temp_distance[intIndex];

	return res;
}

AbstractGraph::path* C11ThreadsVersion::getCriticalPath() {
	return getCriticalPath(0);
}

void C11ThreadsVersion::bellmanFord(const int threadIndex, unsigned row) {

	for (int j = 0; j < vertexesNumber; j++) {
		if (matrix[threadIndex][j] != 0) {
			mtxDistanceRuntime.lock();
			if (distance[j] > distance[threadIndex] + matrix[threadIndex][j]) {
				distance[j] = distance[threadIndex] + matrix[threadIndex][j];
			mtxDistanceRuntime.unlock();
			}
		}
	}
}