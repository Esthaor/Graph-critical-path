#include"../include/Incremental.h"

Incremental::Incremental(std::string graphFilename, unsigned vertexesNumber) {
	this->init(graphFilename, vertexesNumber);
}

std::vector<unsigned> Incremental::getCriticalPath() {
	std::vector<unsigned> criticalPath;
	//todo
	return criticalPath;
}