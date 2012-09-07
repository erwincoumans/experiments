#ifndef OPENGL3_CORE_RENDERER_H
#define OPENGL3_CORE_RENDERER_H

class btDiscreteDynamicsWorld;
class GLInstancingRenderer;

class OpenGL3CoreRenderer
{

	GLInstancingRenderer* m_instanceRenderer;
public:
	OpenGL3CoreRenderer();
	virtual ~OpenGL3CoreRenderer();
	void init();
	void reshape(int w, int h); 
	void keyboardCallback(unsigned char key);
	void renderPhysicsWorld(const btDiscreteDynamicsWorld* world);

};

#endif //OPENGL3_CORE_RENDERER_H

