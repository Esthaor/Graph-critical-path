#pragma once
#include "AbstractGraph.h"
#include <thread>
#include <mutex>
#include <algorithm>
#include <iterator>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

class CUDAVersion: public AbstractGraph {

	int blocksNumber = 1;
	int threadsNumber = 32;

public:
	CUDAVersion(std::string graphFilename, unsigned vertexesNumber);
	virtual AbstractGraph::path* getCriticalPath(unsigned vertexStart) override;
	virtual AbstractGraph::path* getCriticalPath() override;
	void bellmanFord(unsigned row, std::pair<std::vector<long>, std::vector<unsigned>>* pair);

	virtual bool linearMatrix() override { return true; };
};