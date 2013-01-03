#ifndef OPENCL_INIT_DEMO_H
#define OPENCL_INIT_DEMO_H
#include "DemoBase.h"

class OpenCLInitDemo : public DemoBase
{
public:
	
				
	static DemoBase* CreateFunc();

	virtual void step();

};

#endif

