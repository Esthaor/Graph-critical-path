#include"AbstractGraph.h"

class Incremental : public AbstractGraph {

	public:
		virtual std::vector<unsigned> getCriticalPath() override;

};