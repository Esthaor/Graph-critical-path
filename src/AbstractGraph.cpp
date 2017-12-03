#include"../include/AbstractGraph.h"

void AbstractGraph::init(std::string graphFilename, unsigned vertexesNumber) {

	std::cout << "AbstractGraph::AbstractGraph" << std::endl;
		
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
	std::string line, edgeValue;
	int row_idx = 0, col_idx = 0;

	if (graphFile.is_open()) {
		
		while (std::getline(graphFile, line)) {

			std::istringstream line_stream(line);

			while (std::getline(line_stream, edgeValue, ' ')) {
				matrix[row_idx][col_idx] = std::stoul(edgeValue);
				col_idx++;
			}

			col_idx = 0;
			row_idx++;

		}

	} else {
		std::cout << "Couldn't open graph file" << std::endl;
	}

}

void AbstractGraph::printMatrix() {
	for (unsigned row = 0; row < vertexesNumber; row++) {
		for (unsigned col = 0; col < vertexesNumber; col++) {
			std::cout << " " << matrix[row][col];
		}
		std::cout << std::endl;
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
