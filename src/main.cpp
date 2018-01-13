#include<iostream>
#include"../include/Incremental.h"
#include"../include/C11ThreadsVersion.h"


const bool TEST_MODE = false;
const bool SAVE_MODE = false;
const unsigned VERTEXES = 5;
const std::string GRAPH_FILE = "gen/graph.txt";

void printVector(std::vector<unsigned> v) {
//	std::ostringstream oss;
//	std::copy(v.begin(), v.end() - 1, std::ostream_iterator<int>(oss, ","));	// Convert all but the last element to avoid a trailing ","
//	oss << v.back();	// Now add the last element with no delimiter
//	std::cout << "Vertexes: " << oss.str() << std::endl;
	std::cout << "isEmpty: " << v.empty() << std::endl;
	std::cout << "Vertexes: ";
	for (auto i = v.begin(); i != v.end(); ++i) std::cout << *i << ' ';
	std::cout << std::endl;
}

void incremental(std::string graphFilename, unsigned vertexesNumber) {
	Incremental* incremental = new Incremental(graphFilename, vertexesNumber);
	
	if (TEST_MODE) {
		incremental->printMatrix();
	}

	if (SAVE_MODE) {
		incremental->saveMatrix();
	}

	clock_t t = clock();
	Incremental::path* path = incremental->getCriticalPath();
	t = clock() - t;

	std::cout << "pathLength: " << path->pathLength << std::endl;

	//printVector(path->vertexes);

	std::cout << "Calculated in: " << t << "[ms]" << std::endl;

	delete incremental;
	delete path;
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
	C11ThreadsVersion::path* path = threadsVersion->getCriticalPath();
	t = clock() - t;

	std::cout << "pathLength: " << path->pathLength << std::endl;

	//printVector(path->vertexes);

	std::cout << "Calculated in: " << t << "[ms]" << std::endl;

	delete threadsVersion;
	delete path;
}


int main(int argc, char** argv) {

	if (argc > 1) {
		incremental(GRAPH_FILE, std::stoul(argv[1])); // default mode
	}
	else {
		incremental(GRAPH_FILE, VERTEXES); // test mode
		//parallelC11Threads(GRAPH_FILE, VERTEXES); // test mode
	}
	std::cin.get();

}