#include"../include/C11ThreadsVersion.h"

C11ThreadsVersion::C11ThreadsVersion(std::string graphFilename, unsigned vertexesNumber) {
	this->init(graphFilename, vertexesNumber);
}

AbstractGraph::path * C11ThreadsVersion::getCriticalPath(unsigned vertexStart)
{
	return nullptr;
}

AbstractGraph::path* C11ThreadsVersion::getCriticalPath() {
	for (int i = 0; i < vertexesNumber; i++)
		for (int j = 0; j < vertexesNumber; j++)
			matrix[i][j] = -matrix[i][j];

	path* res = new path();

	std::thread* threads = new std::thread[vertexesNumber];
	for (int i = 0; i < vertexesNumber; i++) {
		threads[i] = std::thread(&C11ThreadsVersion::parallelCode, this, i);
		threads[i].join();
	}

	//res->pathLength = -min;
	//res->pathStart = minStart;
	//res->pathEnd = minEnd;
	return res;
}

void C11ThreadsVersion::parallelCode(int i) {
	std::vector<long> longest = bellmanFord(i);
	int intIndex = std::min_element(longest.begin(), longest.end()) - longest.begin();
	int pathLength = longest[intIndex];

	mtx.lock();
	if (pathLength < min) {
		min = pathLength;
		minStart = i;
		minEnd = intIndex;
	}
	mtx.unlock();

}

std::vector<long> C11ThreadsVersion::bellmanFord(unsigned row) {
	std::vector<long> distance;
	std::vector<unsigned> predecessor;

	for (int i = 0; i < vertexesNumber; i++) {
		distance.push_back(LONG_MAX);
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


/*

 - dla grupy nodów wywo³ywaæ 1 w¹tek
 - partycjonowanie sieci
 - "zbyt drobne ziarno"
 - najd³u¿sza œcie¿ka jeœli znadziemy uzasadnienie or powrót do wierzcho³ek A - wierzcho³ek B

*/