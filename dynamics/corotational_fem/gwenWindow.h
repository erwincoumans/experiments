
#ifndef MY_GWEN_WINDOW_H
#define MY_GWEN_WINDOW_H


#include "Gwen/Gwen.h"
#include "Gwen/Controls/Button.h"
#include "Gwen/Skins/Simple.h"
#include "Gwen/Renderers/OpenGL_DebugFont.h"
#include "Gwen/Input/Windows.h"

extern Gwen::Renderer::OpenGL_DebugFont * pRenderer;
extern Gwen::Skin::Simple skin;
extern Gwen::Controls::Canvas* pCanvas;
extern Gwen::Input::Windows GwenInput;
	

class Application;

void	setupGUI(Application* app,int width, int height);



#endif //MY_GWEN_WINDOW_H
