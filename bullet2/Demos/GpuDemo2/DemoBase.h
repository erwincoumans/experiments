#ifndef DEMO_BASE_H
#define DEMO_BASE_H

class DemoBase
{
	public:
	typedef DemoBase* (CreateFunc)();

	virtual ~DemoBase(){}
	virtual void step()=0;

};

#endif//DEMO_BASE_H

