#include<iostream>
#include"Incremental.h"

std::string GRAPH_FILE = "graph.txt";

void incremental(std::string graphFilename, unsigned vertexesNumber) {
	Incremental* incremental = new Incremental(graphFilename, vertexesNumber);
	std::vector<unsigned> criticalPath = incremental->getCriticalPath();
}


int main(int argc, char** argv) {

	if (argc > 1) {
		incremental(GRAPH_FILE, std::stoul(argv[1])); // default mode
	}

}