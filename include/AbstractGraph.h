#include<iostream>
#include<vector>

class AbstractGraph {

	unsigned** matrix;

	inline unsigned** getMatrix();
	inline unsigned getValueFromMatrix(unsigned x, unsigned y);

	public:
		virtual 
		virtual ~AbstractGraph();

		void loadGraphFromFile();
		inline unsigned* getVertexEdges(unsigned vertexNumber);

		virtual std::vector<unsigned> getCriticalPath() = 0;

};