#include"../include/AbstractGraph.h"
#include<algorithm>

class Incremental : public AbstractGraph {

	public:
		Incremental(std::string graphFilename, unsigned vertexesNumber);
		virtual AbstractGraph::path* getCriticalPath() override;
		std::vector<long> bellmanFord(unsigned row);

};

/*
0 0 0 0 0
55 0 0 0 0
204 15 0 0 0
0 142 73 0 0
0 42 0 24 0


*/