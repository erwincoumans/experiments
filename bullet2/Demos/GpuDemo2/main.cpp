

#include "testDemo.h"
#include "OpenCLInitDemo.h"
#include "BroadphaseDemo.h"
#include "ConvexCollisionTest.h"

#include "DemoBase.h"
#include <stdio.h>

DemoBase::CreateFunc* allDemos[] =
{
	//testDemo::CreateFunc,
	//OpenCLInitDemo::CreateFunc,
//	BroadphaseDemo::CreateFunc,
	//CompoundShapeDemo::CreateFunc,
	ConvexCollisionTest::CreateFunc
	
};

int main(int argc, char* argv[])
{
	int numdemos = sizeof(allDemos)/sizeof(DemoBase::CreateFunc*);
	printf("numdemos = %d\n", numdemos);
	
	for (int i=0;i<numdemos;i++)
	{
		DemoBase* demo = allDemos[i]();//testDemo::CreateFunc();
		demo->step();

		delete demo;
	}
	
	return 0;
}

