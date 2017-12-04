#pragma once
#include"AbstractGraph.h"
#include<algorithm>

class Incremental : public AbstractGraph {

	public:
		Incremental(std::string graphFilename, unsigned vertexesNumber);
		virtual AbstractGraph::path* getCriticalPath() override;
		std::vector<long> bellmanFord(unsigned row);

};