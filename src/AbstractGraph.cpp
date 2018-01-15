#include"../include/AbstractGraph.h"

void AbstractGraph::init(std::string graphFilename, unsigned vertexesNumber) {
	usingLinearMatrix = linearMatrix();
	this->graphFilename = graphFilename;
	this->vertexesNumber = vertexesNumber;

	if (usingLinearMatrix) {
		linear_matrix = new int[vertexesNumber * vertexesNumber];

		for (int i = 0; i < vertexesNumber; i++) {
			for (int j = 0; j < vertexesNumber; j++) {
				linear_matrix[j + i * vertexesNumber] = 0;
			}
		}
	}
	else {
		this->matrix = new int*[vertexesNumber];

		for (int i = 0; i < vertexesNumber; i++) {
			this->matrix[i] = new int[vertexesNumber];

			for (int j = 0; j < vertexesNumber; j++) {
				this->matrix[i][j] = 0;
			}
		}
	}

	if (GENERATE_GRAPH) {
		this->generateGraph();
	} else {
		this->loadGraphFromFile(graphFilename);
	}
}

AbstractGraph::~AbstractGraph() {
	if (usingLinearMatrix) {
		delete linear_matrix;
	}
	else {
		for (int i = 0; i < vertexesNumber; i++) {
			delete this->matrix[i];
		}
		delete matrix;
	}
}

void AbstractGraph::loadGraphFromFile(std::string filename) {

	std::ifstream graphFile;
	graphFile.open(filename);
	std::string line, edgeValue;
	int row_idx = 0, col_idx = 0;

	if (graphFile.is_open()) {
		if (usingLinearMatrix) {
			while (std::getline(graphFile, line)) {
				std::istringstream line_stream(line);

				while (std::getline(line_stream, edgeValue, ' ')) {
					linear_matrix[row_idx * vertexesNumber + col_idx] = std::stoi(edgeValue);
					col_idx++;
				}

				col_idx = 0;
				row_idx++;
			}
		}
		else {
			while (std::getline(graphFile, line)) {
				std::istringstream line_stream(line);

				while (std::getline(line_stream, edgeValue, ' ')) {
					matrix[row_idx][col_idx] = std::stoi(edgeValue);
					col_idx++;
				}

				col_idx = 0;
				row_idx++;
			}
		}
	}
	else {
		std::cout << "Couldn't open graph file" << std::endl;
	}

	graphFile.close();
}

void AbstractGraph::generateGraph() {
	std::srand(time(NULL));

	if (usingLinearMatrix) {
		for (int i = 0; i < vertexesNumber; i++) {
			for (int j = 0; j < vertexesNumber; j++) {
				if (j > i) {
					if ((std::rand() % 100) < 44) {
						linear_matrix[j + i * vertexesNumber] = (std::rand() % 511);
					}
					else {
						linear_matrix[j + i * vertexesNumber] = 0;
					}
				}
			}
		}
	}
	else {
		for (int i = 0; i < vertexesNumber; i++) {
			for (int j = 0; j < vertexesNumber; j++) {
				if (j > i) {
					if ((std::rand() % 100) < 44) {
						matrix[i][j] = (std::rand() % 511);
					}
					else {
						matrix[i][j] = 0;
					}
				}
			}
		}
	}
}

int AbstractGraph::rowCount(int y) {
	unsigned  count = 0;
	for (int i = 0; i < vertexesNumber; i++) {
		if (matrix[i][y] != 0) {
			count++;
		}
	}

	return count;
}

int AbstractGraph::colCount(int x) {
	unsigned  count = 0;
	for (int i = 0; i < vertexesNumber; i++) {
		if (matrix[x][i] != 0) {
			count++;
		}
	}

	return count;
}

void AbstractGraph::saveMatrix() {
	std::ofstream graphFile;
	graphFile.open(graphFilename);
	if (graphFile.is_open()) {
		if (usingLinearMatrix) {
			for (unsigned row = 0; row < vertexesNumber; row++) {
				for (unsigned col = 0; col < vertexesNumber; col++) {
					graphFile << " " << linear_matrix[row * vertexesNumber + col];
				}
				graphFile << std::endl;
			}
		}
		else {
			for (unsigned row = 0; row < vertexesNumber; row++) {
				for (unsigned col = 0; col < vertexesNumber; col++) {
					graphFile << " " << matrix[row][col];
				}
				graphFile << std::endl;
			}
		}
	}

	graphFile.close();

}

void AbstractGraph::printMatrix() {
	if (usingLinearMatrix) {
		for (unsigned row = 0; row < vertexesNumber; row++) {
			for (unsigned col = 0; col < vertexesNumber; col++) {
				std::cout << matrix[row * vertexesNumber + col] << " ";
			}
			std::cout << std::endl;
		}
	}
	else {
		for (unsigned row = 0; row < vertexesNumber; row++) {
			for (unsigned col = 0; col < vertexesNumber; col++) {
				std::cout << matrix[row][col] << " ";
			}
			std::cout << std::endl;
		}
	}
}

inline int** AbstractGraph::getMatrix() {
	return matrix;
}

inline int AbstractGraph::getValueFromMatrix(unsigned x, unsigned y) {
	return matrix[x][y];
}

inline int* AbstractGraph::getVertexEdges(unsigned vertexNumber) {
	return matrix[vertexNumber];
}