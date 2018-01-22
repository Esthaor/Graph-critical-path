#pragma once
#include"AbstractGraph.h"
#include<algorithm>
#include<iterator>

using edges = std::vector<std::pair<int, int>*>;

class IncrementalAdjacencyTable : public AbstractGraph {

	edges* adjacency_table;
	std::pair<int, int>*** tab;
	int* tab_sizes;

	void fillAdjacencyTable();

public:
	IncrementalAdjacencyTable(std::string graphFilename, unsigned vertexesNumber);
	virtual AbstractGraph::path* getCriticalPath(unsigned vertexStart) override;
	virtual AbstractGraph::path* getCriticalPath() override;
	std::pair<std::vector<int>, std::vector<unsigned>>* bellmanFord(unsigned row);

	virtual bool linearMatrix() override { return false; };
};