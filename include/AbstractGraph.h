#pragma once
#include<iostream>
#include<vector>
#include<fstream>
#include<sstream>
#include<string>
#include<ctime>

class AbstractGraph {
	bool GENERATE_GRAPH = false;
protected:
	int** matrix;

	std::string graphFilename;
	unsigned vertexesNumber;

	inline int** getMatrix();
	inline int getValueFromMatrix(unsigned x, unsigned y);

	int rowCount(int y);
	int colCount(int x);

public:
	struct path {
		int pathLength;
		int pathStart;
		int pathEnd;
	};

	void init(std::string graphFilename, unsigned vertexesNumber);
	virtual ~AbstractGraph();

	void loadGraphFromFile(std::string filename);
	void generateGraph();
	inline int* getVertexEdges(unsigned vertexNumber);

	void printMatrix();
	void saveMatrix();

	virtual AbstractGraph::path* getCriticalPath() = 0;

};