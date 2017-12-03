#include"AbstractGraph.h"

AbstractGraph::AbstractGraph(std::string graphFilename, unsigned vertexesNumber) {
	this->graphFilename = graphFilename;
	this->vertexesNumber = vertexesNumber;

	this->matrix = new unsigned*[vertexesNumber];

	for (int i = 0; i < vertexesNumber; i++) {
		this->matrix[i] = new unsigned[vertexesNumber];
	}

	this->loadGraphFromFile(graphFilename);
}

AbstractGraph::~AbstractGraph() {

	for (int i = 0; i < vertexesNumber; i++) {
		delete this->matrix[i];
	}
	delete matrix;
}

void AbstractGraph::loadGraphFromFile(std::string filename) {

	std::ifstream graphFile;
	graphFile.open(filename);
	std::string edgeValue;
	int row_idx = 0, col_idx = 0;

	if (graphFile.is_open()) {
		
		while (std::getline(graphFile, edgeValue, ' ')) {

			if (edgeValue == "\n") {
				col_idx = 0;
				row_idx++;
			} else {
				matrix[row_idx][col_idx] = std::stoul(edgeValue);
			}

			/*
				if (row_idx > vertexesNumber) {
					std::cout << "wrong indexes" << std::endl;
				}
			*/

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
