#include"AbstractGraph.h"


AbstractGraph::~AbstractGraph() {
	delete matrix;
}

void AbstractGraph::loadGraphFromFile() {
	// todo
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