#include<iostream>
#include"../include/Incremental.h"

unsigned VERTEXES = 10;
std::string GRAPH_FILE = "gen/graph.txt";

void incremental(std::string graphFilename, unsigned vertexesNumber) {
	Incremental* incremental = new Incremental(graphFilename, vertexesNumber);

	incremental->printMatrix();

	// std::vector<unsigned> criticalPath = incremental->getCriticalPath();
}


int main(int argc, char** argv) {

	if (argc > 1) {
		incremental(GRAPH_FILE, std::stoul(argv[1])); // default mode
	}
	else {
		incremental(GRAPH_FILE, VERTEXES); // test mode
	}
	std::cin.get();
}