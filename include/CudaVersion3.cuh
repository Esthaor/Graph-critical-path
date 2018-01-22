#pragma once
#include "AbstractGraph.h"
#include <thread>
#include <mutex>
#include <algorithm>
#include <iterator>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define VERTEXES 10000

using edges = std::vector<std::pair<int, int>*>;

class CudaVersion3 : public AbstractGraph {

	//const int vertexes = 10000;
	int blocksNumber;
	int threadsNumber;

	edges* adjacency_table;
	std::pair<int, int>** tab;
	std::pair<int, int> stab[VERTEXES * VERTEXES];

	//std::vector<int> tab_end[VERTEXES];

	int* tab_sizes;

	cudaEvent_t start, stop;
	float miliseconds = 0;

	void fillAdjacencyTable();

public:
	CudaVersion3(std::string graphFilename, unsigned vertexesNumber);
	virtual AbstractGraph::path* getCriticalPath(unsigned vertexStart) override;
	virtual AbstractGraph::path* getCriticalPath() override;
	void bf(unsigned row, std::pair<std::vector<int>, std::vector<unsigned>>* pair);

	float getMiliseconds() { return miliseconds; };
	int getBlocksNumber() { return blocksNumber; };
	int getThreadsNumber() { return threadsNumber; };

	virtual bool linearMatrix() override { return false; };



};