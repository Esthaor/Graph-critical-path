#include"../include/AbstractGraph.h"

class Incremental : public AbstractGraph {

	public:
		Incremental(std::string graphFilename, unsigned vertexesNumber);
		virtual std::vector<unsigned> getCriticalPath() override;

};