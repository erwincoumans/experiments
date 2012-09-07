
#ifndef GLES2_RENDERER_H
#define GLES2_RENDERER_H

class btDiscreteDynamicsWorld;
class GLES2ShapeDrawer;

class GLES2Renderer
{

	GLES2ShapeDrawer* m_shapeDrawer;
	int m_debugMode;

	void	renderscene(int modelMatrix, const btDiscreteDynamicsWorld* world);

		void stepLeft();
	void stepRight();
	void stepFront();
	void stepBack();
	void zoomIn();
	void zoomOut();

public:
		
		GLES2Renderer();
		virtual ~GLES2Renderer();

		virtual void init(int screenWidth, int screenHeight);

		void draw(const btDiscreteDynamicsWorld* world,  int width, int height);
		int getDebugMode() const
		{
			return m_debugMode;
		}
		void updateCamera();
		void keyboardCallback(unsigned char key);
};
#endif //GLES2_RENDERER_H
