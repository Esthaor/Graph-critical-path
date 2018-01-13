#pragma once
#include "AbstractGraph.h"
#include <thread>
#include <mutex>
#include <algorithm>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

class CUDAVersion: public AbstractGraph {

	const unsigned blocksNumber = 1;
	const unsigned threadsNumber = 32;

	__global__ void kernel(int vertexesNumber, long** matrix, long* distance);

public:
	CUDAVersion(std::string graphFilename, unsigned vertexesNumber);
	virtual AbstractGraph::path * CUDAVersion::getCriticalPath(unsigned vertexStart) override;
	virtual AbstractGraph::path* getCriticalPath() override;
	void bellmanFord(unsigned row, std::pair<std::vector<long>, std::vector<unsigned>>* pair);
	  
};