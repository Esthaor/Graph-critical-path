#pragma once
#include "AbstractGraph.h"
#include "omp.h"
#include <string>
#include <algorithm>

class OpenMPVersion : public AbstractGraph {

public:
	OpenMPVersion(std::string graphFilename, unsigned vertexesNumber);
	//~OpenMPVersion();
	virtual AbstractGraph::path* getCriticalPath(unsigned vertexStart) override;
	virtual AbstractGraph::path* getCriticalPath() override;
	void bellmanFord(unsigned row, std::pair<std::vector<long>, std::vector<unsigned>>* pair);

};

