#ifndef _BT_DEMOS_COMMON_H
#define _BT_DEMOS_COMMON_H
struct DemoApplication
{
	virtual~DemoApplication()
	{
	}

	void myinit()
	{
	}

	float getDeltaTimeInSeconds()
	{
		return 1./60.f;
	}
	void renderme()
	{
	}

	void swapBuffers()
	{

	}

	void setTexturing(bool enable)
	{

	}
	void setShadows(bool enable)
	{
	}

	void clearScreen()
	{
//			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 

	}

	virtual void setCameraDistance(float distance)
	{

	}
	//getDeltaTimeMicroseconds
	//clear screen
	//renderme
	//glFlush();
	//swapBuffers();
	//enableTexturing
	//enableShadows
	//setCameraDistance

};

#endif //_BT_DEMOS_COMMON_H
