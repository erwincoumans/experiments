#include "testDemo.h"
#include <stdio.h>

DemoBase* testDemo::CreateFunc()
{
	DemoBase* demo = new testDemo();
	return demo;
}

void testDemo::step()
{
	printf("testDemo::step\n");
}