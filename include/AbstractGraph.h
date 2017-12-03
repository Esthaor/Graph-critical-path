#include<iostream>
#include<vector>
#include<fstream>
#include<sstream>
#include<string>
#include<ctime>

class AbstractGraph {

protected:
	unsigned** matrix;

	std::string graphFilename;
	unsigned vertexesNumber;

	inline unsigned** getMatrix();
	inline unsigned getValueFromMatrix(unsigned x, unsigned y);

public:
	void init(std::string graphFilename, unsigned vertexesNumber);
	virtual ~AbstractGraph();

	void loadGraphFromFile(std::string filename);
	inline unsigned* getVertexEdges(unsigned vertexNumber);

	void printMatrix();

	virtual std::vector<unsigned> getCriticalPath() = 0;

};