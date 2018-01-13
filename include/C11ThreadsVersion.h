#pragma once
#include"AbstractGraph.h"
#include<thread>
#include<mutex>
#include<algorithm>

class C11ThreadsVersion : public AbstractGraph {
protected:
	std::mutex mtx;

	int min = 0;
	int minStart = 0;
	int minEnd = 0;

	void parallelCode(int i);
public:
	C11ThreadsVersion(std::string graphFilename, unsigned vertexesNumber);
	virtual AbstractGraph::path* getCriticalPath(unsigned vertexStart) override;
	virtual AbstractGraph::path* getCriticalPath() override;
	std::vector<long> bellmanFord(unsigned row);
};

