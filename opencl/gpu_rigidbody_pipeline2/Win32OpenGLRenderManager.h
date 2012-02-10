
#ifndef _WIN32_OPENGL_RENDER_MANAGER_H
#define _WIN32_OPENGL_RENDER_MANAGER_H


#define RM_DECLARE_HANDLE(name) typedef struct name##__ { int unused; } *name

RM_DECLARE_HANDLE(RenderObjectHandle);

struct InternalData2;

class Win32OpenGLWindow
{
	protected:
		
		struct InternalData2*	m_data;
		
		void enableOpenGL();
		
		void disableOpenGL();

		void pumpMessage();
	
		

public:

	Win32OpenGLWindow();

	virtual ~Win32OpenGLWindow();

	virtual	void	init(); //default implementation uses default settings for width/height/fullscreen

	void	init(int width,int height, bool fullscreen=false, int colorBitsPerPixel=0, void* windowHandle=0);
	
	void	switchFullScreen(bool fullscreen,int width=0,int height=0,int colorBitsPerPixel=0);

	virtual	void	exit();


	virtual	void	startRendering();

	virtual	void	renderAllObjects();

	virtual	void	endRendering();

	virtual	float	getTimeInSeconds();

	virtual void	setDebugMessage(int x,int y,const char* message);
	
	virtual bool requestedExit();

};

#endif //_WIN32_OPENGL_RENDER_MANAGER_H
