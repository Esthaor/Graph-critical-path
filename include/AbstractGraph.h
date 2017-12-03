#include<iostream>
#include<vector>
#include<fstream>
#include<string>
#include<ctime>

class AbstractGraph {

	unsigned** matrix;

	inline unsigned** getMatrix();
	inline unsigned getValueFromMatrix(unsigned x, unsigned y);

	public:
		virtual ~AbstractGraph();

		void loadGraphFromFile(std::string filename);
		inline unsigned* getVertexEdges(unsigned vertexNumber);

		virtual std::vector<unsigned> getCriticalPath() = 0;

};