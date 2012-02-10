#ifndef GLUT_RENDERER_H
#define GLUT_RENDERER_H

#include "btGlutInclude.h"
#include "LinearMath/btVector3.h"

struct GlutRenderer
{
	static GlutRenderer* gDemoApplication;
	int m_glutScreenWidth;
	int m_glutScreenHeight;

	btVector3 m_cameraPosition;
	btVector3 m_cameraTargetPosition;
	btScalar m_cameraDistance;
	btVector3 m_cameraUp;
	float m_azimuth;
	float m_elevation;


	GlutRenderer(int argc, char* argv[]);
	
	virtual void initGraphics(int width, int height);
	virtual void cleanup() {}
	
	void runMainLoop();

	virtual void updateScene(){};
	
	virtual void renderScene();

	virtual void	keyboardCallback(unsigned char key, int x, int y) {};
	virtual void	keyboardUpCallback(unsigned char key, int x, int y) {}
	virtual void	specialKeyboard(int key, int x, int y){}
	virtual void	specialKeyboardUp(int key, int x, int y){}
	virtual void	resize(int w, int h);
	virtual void	mouseFunc(int button, int state, int x, int y);
	virtual void	mouseMotionFunc(int x,int y);
	virtual void displayCallback();
	

};

#endif //GLUT_RENDERER_H
