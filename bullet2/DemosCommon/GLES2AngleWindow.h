
#ifndef GLES2_ANGLE_WINDOW_H
#define GLES2_ANGLE_WINDOW_H


#include "../../rendering/rendertest/Win32Window.h"

struct GLES2Context;

class GLES2AngleWindow : public Win32Window
{
	bool m_OpenGLInitialized;

	protected:
		
		
	bool enableGLES2();
		
	void disableGLES2();

	GLES2Context* m_esContext;
		
	
public:

	GLES2AngleWindow();

	virtual ~GLES2AngleWindow();

	virtual	void	createWindow(const btgWindowConstructionInfo& ci);
	

	virtual	void	closeWindow();


	virtual	void	startRendering();

	virtual	void	renderAllObjects();

	virtual	void	endRendering();
	


};

#endif //GLES2_ANGLE_WINDOW_H
