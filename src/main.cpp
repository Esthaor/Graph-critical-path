#include<iostream>
#include"../include/Incremental.h"
#include"../include/C11ThreadsVersion.h"


const bool TEST_MODE = false;
const bool SAVE_MODE = false;
const unsigned VERTEXES = 250;
const std::string GRAPH_FILE = "gen/graph.txt";

void incremental(std::string graphFilename, unsigned vertexesNumber) {
	Incremental* incremental = new Incremental(graphFilename, vertexesNumber);
	
	if (TEST_MODE) {
		incremental->printMatrix();
	}

	if (SAVE_MODE) {
		incremental->saveMatrix();
	}

	clock_t t = clock();
	Incremental::path* minValues = incremental->getCriticalPath();
	t = clock() - t;

	std::cout << "pathLength: " << minValues->pathLength << std::endl;
	std::cout << "pathStart: " << minValues->pathStart << std::endl;
	std::cout << "pathEnd: " << minValues->pathEnd << std::endl;
	std::cout << "Calculated in: " << t << "[ms]" << std::endl;

	delete incremental;
	delete minValues;
}

void parallelC11Threads(std::string graphFilename, unsigned vertexesNumber) {
	C11ThreadsVersion* threadsVersion = new C11ThreadsVersion(graphFilename, vertexesNumber);

	if (TEST_MODE) {
		threadsVersion->printMatrix();
	}

	if (SAVE_MODE) {
		threadsVersion->saveMatrix();
	}

	clock_t t = clock();
	C11ThreadsVersion::path* minValues = threadsVersion->getCriticalPath();
	t = clock() - t;

	std::cout << "pathLength: " << minValues->pathLength << std::endl;
	std::cout << "pathStart: " << minValues->pathStart << std::endl;
	std::cout << "pathEnd: " << minValues->pathEnd << std::endl;
	std::cout << "Calculated in: " << t << "[ms]" << std::endl;

	delete threadsVersion;
	delete minValues;
}


int main(int argc, char** argv) {

	if (argc > 1) {
		incremental(GRAPH_FILE, std::stoul(argv[1])); // default mode
	}
	else {
		//incremental(GRAPH_FILE, VERTEXES); // test mode
		parallelC11Threads(GRAPH_FILE, VERTEXES); // test mode
	}
	std::cin.get();

}