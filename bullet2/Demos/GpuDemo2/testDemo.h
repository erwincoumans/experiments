#ifndef TEST_DEMO_H
#define TEST_DEMO_H
#include "DemoBase.h"

class testDemo : public DemoBase
{
	public:
	static DemoBase* CreateFunc();
	
	virtual void step();
};

#endif //TEST_DEMO_H

