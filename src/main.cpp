#include<iostream>
#include"Incremental.h"

void incremental() {
	Incremental* incremental = new Incremental();
	incremental->loadGraphFromFile();

	std::vector<unsigned> criticalPath = incremental->getCriticalPath();

}


int main(int argc, char** argv) {

	if (argc == 1) {
		incremental(); // default mode
	}
	else {
		// todo: analazy argv for mprogram mode -> paralel; cuda_parallel
	}


}