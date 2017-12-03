#include"AbstractGraph.h"

AbstractGraph::~AbstractGraph() {
	delete matrix;
}

void AbstractGraph::loadGraphFromFile(std::string filename) {
	std::ifstream graphFile;
	graphFile.open(filename);
	std::string v;
	int rowIndex = 0;

	if (graphFile.is_open()) {
		while (std::getline(graphFile, v, ',')) {
			for (unsigned i = 0; i < v.length(); i++) {
				if (isdigit(v[i])) {
					// write to matrix
				}

			}
		}
	} else {
		std::cout << "Couldn't open graph file" << std::endl;
	}

}

inline unsigned** AbstractGraph::getMatrix() {
	return matrix;
}

inline unsigned AbstractGraph::getValueFromMatrix(unsigned x, unsigned y) {
	return matrix[x][y];
}

inline unsigned* AbstractGraph::getVertexEdges(unsigned vertexNumber) {
	return matrix[vertexNumber];
}
