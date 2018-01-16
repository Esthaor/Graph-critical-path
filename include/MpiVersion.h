#pragma once
#include "AbstractGraph.h"
#include <string>
#include <algorithm>

class MpiVersion : public AbstractGraph {
public:
	MpiVersion(std::string graphFilename, unsigned vertexesNumber);
	virtual AbstractGraph::path* getCriticalPath(unsigned vertexStart) override;
	virtual AbstractGraph::path* getCriticalPath() override;
	std::pair<std::vector<long>, std::vector<unsigned>>* bellmanFord(unsigned row);

	virtual bool linearMatrix() override { return true; };
};

