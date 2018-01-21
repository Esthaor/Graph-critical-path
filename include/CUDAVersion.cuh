#pragma once
#include "AbstractGraph.h"
#include <thread>
#include <mutex>
#include <algorithm>
#include <iterator>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

class CUDAVersion: public AbstractGraph {

	int blocksNumber;
	int threadsNumber;

	cudaEvent_t start, stop;
	float miliseconds = 0;

public:
	CUDAVersion(std::string graphFilename, unsigned vertexesNumber);
	virtual AbstractGraph::path* getCriticalPath(unsigned vertexStart) override;
	virtual AbstractGraph::path* getCriticalPath() override;
	void bellmanFord(unsigned row, std::pair<std::vector<int>, std::vector<unsigned>>* pair);
	void bf(unsigned row, std::pair<std::vector<int>, std::vector<unsigned>>* pair);

	float getMiliseconds() { return miliseconds; };
	int getBlocksNumber() { return blocksNumber; };
	int getThreadsNumber() { return threadsNumber; };

	virtual bool linearMatrix() override { return true; };
};