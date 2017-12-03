#include<iostream>
#include<vector>
#include<fstream>
#include<sstream>
#include<string>
#include<ctime>

class AbstractGraph {

protected:
	int** matrix;

	std::string graphFilename;
	unsigned vertexesNumber;

	inline int** getMatrix();
	inline int getValueFromMatrix(unsigned x, unsigned y);

public:
	struct path {
		int pathLength;
		int pathStart;
		int pathEnd;
	};

	void init(std::string graphFilename, unsigned vertexesNumber);
	virtual ~AbstractGraph();

	void loadGraphFromFile(std::string filename);
	inline int* getVertexEdges(unsigned vertexNumber);

	void printMatrix();

	virtual AbstractGraph::path* getCriticalPath() = 0;

};