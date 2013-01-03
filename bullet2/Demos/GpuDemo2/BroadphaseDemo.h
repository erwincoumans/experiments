#ifndef BROADPHASE_DEMO_H
#define BROADPHASE_DEMO_H

#include "DemoBase.h"
class BroadphaseDemo : public DemoBase
{
public:
	static DemoBase* CreateFunc();

	virtual void step();
};
#endif

