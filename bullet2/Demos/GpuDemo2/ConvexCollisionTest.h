#ifndef CONVEX_COLLISION_TEST_H
#define CONVEX_COLLISION_TEST_H

#include "DemoBase.h"

class ConvexCollisionTest : public DemoBase
{
	public:

	static DemoBase* CreateFunc();

	virtual void step();

};
#endif //CONVEX_COLLISION_TEST_H
