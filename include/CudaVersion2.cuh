#pragma once
#include "AbstractGraph.h"
#include <thread>
#include <mutex>
#include <algorithm>
#include <iterator>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using edges = std::vector<std::pair<int, int>*>;

class CudaVersion2 : public AbstractGraph {
	int* tab_sizes;

	cudaEvent_t start, stop;
	float miliseconds = 0;

	void countEdges();

public:
	CudaVersion2(std::string graphFilename, unsigned vertexesNumber);
	
	virtual AbstractGraph::path* getCriticalPath(unsigned vertexStart) override;
	virtual AbstractGraph::path* getCriticalPath() override;
	virtual bool linearMatrix() override { return true; };
	
	void bf(unsigned row, std::pair<std::vector<int>, std::vector<unsigned>>* pair);
	float getMiliseconds() { return miliseconds; };

};