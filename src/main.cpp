#include<iostream>
#include"Incremental.h"

std::string GRAPH_FILE = "graph.txt";

void incremental() {
	Incremental* incremental = new Incremental();
	incremental->loadGraphFromFile(GRAPH_FILE);

	std::vector<unsigned> criticalPath = incremental->getCriticalPath();

}


int main(int argc, char** argv) {

	if (argc == 1) {
		incremental(); // default mode
	}
	else {
		// todo: analyze argv for mprogram mode -> paralel; cuda_parallel
	}


}