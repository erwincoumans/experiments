#ifndef _BT_OPENGL2_RENDERER_H
#define _BT_OPENGL2_RENDERER_H

class btDiscreteDynamicsWorld;
class GL_ShapeDrawer;

#include"LinearMath/btVector3.h"

class OpenGL2Renderer
{
	float	m_cameraDistance;
	int	m_debugMode;
	
	float m_ele;
	float m_azi;
	btVector3 m_cameraPosition;
	btVector3 m_cameraTargetPosition;//look at

	int	m_mouseOldX;
	int	m_mouseOldY;
	int	m_mouseButtons;
public:
	int	m_modifierKeys;
protected:

	float m_scaleBottom;
	float m_scaleFactor;
	btVector3 m_cameraUp;
	int	m_forwardAxis;
	float m_zoomStepSize;

	int m_openglViewportWidth;
	int m_openglViewportHeight;

	btVector3 m_sundirection;

	float	m_frustumZNear;
	float	m_frustumZFar;

	bool m_enableshadows;
	GL_ShapeDrawer*	m_shapeDrawer;

	int	m_ortho;
	void	renderscene(int pass,const btDiscreteDynamicsWorld* world);

	void stepLeft();
	void stepRight();
	void stepFront();
	void stepBack();
	void zoomIn();
	void zoomOut();

public:
	OpenGL2Renderer();
	virtual ~OpenGL2Renderer();
	void init();
	void updateCamera();
	int getDebugMode() const { return m_debugMode;}
	void setOrthographicProjection();
	void resetPerspectiveProjection() ;
	void reshape(int w, int h); 
	void keyboardCallback(unsigned char key);
	void renderPhysicsWorld(const btDiscreteDynamicsWorld* world);


};

#endif //_BT_OPENGL2_RENDERER_H
