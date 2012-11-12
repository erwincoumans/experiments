

#include "Gwen/Gwen.h"
#include "Gwen/Skins/Simple.h"

#include "UnitTest.h"


#include "gl/glew.h"

#ifdef __APPLE__
#include "MacOpenGLWindow.h"
#else
#include "Win32OpenGLWindow.h"
#endif

#include "../OpenGLTrueTypeFont/opengl_fontstashcallbacks.h"

#include "GwenOpenGL3CoreRenderer.h"
#include "GLPrimitiveRenderer.h"
#include <assert.h>

Gwen::Controls::Canvas*		pCanvas  = NULL;

void MyMouseMoveCallback( float x, float y)
{
	//btDefaultMouseCallback(button,state,x,y);

	static int m_lastmousepos[2] = {0,0};
	static bool isInitialized = false;
	if (pCanvas)
	{
		if (!isInitialized)
		{
			isInitialized = true;
			m_lastmousepos[0] = x+1;
			m_lastmousepos[1] = y+1;
		}
		bool handled = pCanvas->InputMouseMoved(x,y,m_lastmousepos[0],m_lastmousepos[1]);
	}
}

void MyMouseButtonCallback(int button, int state, float x, float y)
{
	//btDefaultMouseCallback(button,state,x,y);

	if (pCanvas)
	{
		bool handled = pCanvas->InputMouseMoved(x,y,x, y);

		if (button>=0)
		{
			handled = pCanvas->InputMouseButton(button,state);
			if (handled)
			{
				if (!state)
					return;
			}
		}
	}
}

int sWidth = 1500;
int sHeight = 900;
/*
LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	switch (message)
	{
		case WM_SIZE:													// Size Action Has Taken Place

			switch (wParam)												// Evaluate Size Action
			{
				case SIZE_MINIMIZED:									// Was Window Minimized?
				return 0;												// Return

				case SIZE_RESTORED:										// Was Window Restored?
				case SIZE_MAXIMIZED:									// Was Window Maximized?
					sWidth = LOWORD (lParam);
					sHeight = HIWORD (lParam);
					if (pCanvas )
					{
						pCanvas->SetSize(sWidth,sHeight);
						RECT r;
						if ( GetClientRect( hWnd, &r ) )
						{
							glMatrixMode( GL_PROJECTION );
							glLoadIdentity();
							glOrtho( r.left, r.right, r.bottom, r.top, -1.0, 1.0);
							glMatrixMode( GL_MODELVIEW );
							glViewport(0, 0, r.right - r.left, r.bottom - r.top);
						}
					}

				return 0;												// Return
			}
		break;	
	default:
		return DefWindowProc( hWnd, message, wParam, lParam );
			
	}
	return 0;
}


HWND CreateGameWindow( void )
{
	WNDCLASS	wc;
	ZeroMemory( &wc, sizeof( wc ) );

	wc.style			= CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
	wc.lpfnWndProc		= WndProc;
	wc.hInstance		= GetModuleHandle(NULL);
	wc.lpszClassName	= L"GWENWindow";
	wc.hCursor			= LoadCursor( NULL, IDC_ARROW );

	RegisterClass( &wc );


	HWND hWindow = CreateWindowEx( (WS_EX_APPWINDOW | WS_EX_WINDOWEDGE) , wc.lpszClassName, L"GWEN - OpenGL Sample (Using embedded debug font renderer)", (WS_OVERLAPPEDWINDOW | WS_CLIPSIBLINGS | WS_CLIPCHILDREN), -1, -1, 1004, 525, NULL, NULL, GetModuleHandle(NULL), NULL );

	ShowWindow( hWindow, SW_SHOW );
	SetForegroundWindow( hWindow );
	SetFocus( hWindow );

	return hWindow;
}


HGLRC CreateOpenGLDeviceContext()
{
	PIXELFORMATDESCRIPTOR pfd = { 0 };
	pfd.nSize = sizeof( PIXELFORMATDESCRIPTOR );    // just its size
    pfd.nVersion = 1;   // always 1

    pfd.dwFlags = PFD_SUPPORT_OPENGL |  // OpenGL support - not DirectDraw
                  PFD_DOUBLEBUFFER   |  // double buffering support
                  PFD_DRAW_TO_WINDOW;   // draw to the app window, not to a bitmap image

    pfd.iPixelType = PFD_TYPE_RGBA ;    // red, green, blue, alpha for each pixel
    pfd.cColorBits = 24;                // 24 bit == 8 bits for red, 8 for green, 8 for blue.
                                        // This count of color bits EXCLUDES alpha.

    pfd.cDepthBits = 32;                // 32 bits to measure pixel depth.  

	int pixelFormat = ChoosePixelFormat( GetDC( g_pHWND ), &pfd );
    
	if ( pixelFormat == 0 )
    {
        FatalAppExit( NULL, TEXT("ChoosePixelFormat() failed!") );
    }    

	SetPixelFormat( GetDC( g_pHWND ), pixelFormat, &pfd );

	HGLRC OpenGLContext = wglCreateContext( GetDC( g_pHWND ) );
	    
	wglMakeCurrent( GetDC( g_pHWND ), OpenGLContext );

	RECT r;
	if ( GetClientRect( g_pHWND, &r ) )
	{
		glMatrixMode( GL_PROJECTION );
		glLoadIdentity();
		glOrtho( r.left, r.right, r.bottom, r.top, -1.0, 1.0);
		glMatrixMode( GL_MODELVIEW );
		glViewport(0, 0, r.right - r.left, r.bottom - r.top);
	}

	return OpenGLContext;
}

*/

	int droidRegular, droidItalic, droidBold, droidJapanese, dejavu;

sth_stash* initFont()
{
	GLint err;

		struct sth_stash* stash = 0;
	int datasize;
	unsigned char* data;
	float sx,sy,dx,dy,lh;
	GLuint texture;

	stash = sth_create(512,512,OpenGL2UpdateTextureCallback,OpenGL2RenderCallback);//256,256);//,1024);//512,512);
    err = glGetError();
    assert(err==GL_NO_ERROR);
    
	if (!stash)
	{
		fprintf(stderr, "Could not create stash.\n");
		return 0;
	}

	const char* fontPaths[]={
	"./",
	"../../bin/",
	"../bin/",
	"bin/"
	};

	int numPaths=sizeof(fontPaths)/sizeof(char*);
	
	// Load the first truetype font from memory (just because we can).
    
	FILE* fp = 0;
	const char* fontPath ="./";
	char fullFontFileName[1024];

	for (int i=0;i<numPaths;i++)
	{
		
		fontPath = fontPaths[i];
		//sprintf(fullFontFileName,"%s%s",fontPath,"OpenSans.ttf");//"DroidSerif-Regular.ttf");
		sprintf(fullFontFileName,"%s%s",fontPath,"DroidSerif-Regular.ttf");//OpenSans.ttf");//"DroidSerif-Regular.ttf");
		fp = fopen(fullFontFileName, "rb");
		if (fp)
			break;
	}

    err = glGetError();
    assert(err==GL_NO_ERROR);
    
    assert(fp);
    if (fp)
    {
        fseek(fp, 0, SEEK_END);
        datasize = (int)ftell(fp);
        fseek(fp, 0, SEEK_SET);
        data = (unsigned char*)malloc(datasize);
        if (data == NULL)
        {
            assert(0);
            return 0;
        }
        else
            fread(data, 1, datasize, fp);
        fclose(fp);
        fp = 0;
    }
	if (!(droidRegular = sth_add_font_from_memory(stash, data)))
    {
        assert(0);
        return 0;
    }
    err = glGetError();
    assert(err==GL_NO_ERROR);

	// Load the remaining truetype fonts directly.
    sprintf(fullFontFileName,"%s%s",fontPath,"DroidSerif-Italic.ttf");

	if (!(droidItalic = sth_add_font(stash,fullFontFileName)))
	{
        assert(0);
        return 0;
    }
     sprintf(fullFontFileName,"%s%s",fontPath,"DroidSerif-Bold.ttf");

	if (!(droidBold = sth_add_font(stash,fullFontFileName)))
	{
        assert(0);
        return 0;
    }
    err = glGetError();
    assert(err==GL_NO_ERROR);
    
     sprintf(fullFontFileName,"%s%s",fontPath,"DroidSansJapanese.ttf");
    if (!(droidJapanese = sth_add_font(stash,fullFontFileName)))
	{
        assert(0);
        return 0;
    }
    err = glGetError();
    assert(err==GL_NO_ERROR);

	return stash;
}

int main()
{



#ifdef __APPLE__
	MacOpenGLWindow* window = new MacOpenGLWindow();
#else
	Win32OpenGLWindow* window = new Win32OpenGLWindow();
#endif
	btgWindowConstructionInfo wci;
	wci.m_width = sWidth;
	wci.m_height = sHeight;
	
	window->createWindow(wci);
	window->setWindowTitle("render test");
	glewInit();

	sth_stash* font = initFont();

	GLPrimitiveRenderer* primRenderer = new GLPrimitiveRenderer(sWidth,sHeight);
	GwenOpenGL3CoreRenderer* gwenRenderer = new GwenOpenGL3CoreRenderer(primRenderer,font,sWidth,sHeight,1);


	//
	// Create a GWEN OpenGL Renderer
	//
//		Gwen::Renderer::OpenGL_DebugFont * pRenderer = new Gwen::Renderer::OpenGL_DebugFont();

	//
	// Create a GWEN skin
	//
		 

#ifdef USE_TEXTURED_SKIN
	Gwen::Skin::TexturedBase skin;
	skin.SetRender( pRenderer );
	skin.Init("DefaultSkin.png");
#else
	Gwen::Skin::Simple skin;
	skin.SetRender( gwenRenderer );
#endif


	//
	// Create a Canvas (it's root, on which all other GWEN panels are created)
	//
	pCanvas = new Gwen::Controls::Canvas( &skin );
	pCanvas->SetSize( sWidth, sHeight);
	pCanvas->SetDrawBackground( true );
	pCanvas->SetBackgroundColor( Gwen::Color( 150, 170, 170, 255 ) );

	window->setMouseButtonCallback(MyMouseButtonCallback);
	window->setMouseMoveCallback(MyMouseMoveCallback);


	//
	// Create our unittest control (which is a Window with controls in it)
	//
	UnitTest* pUnit = new UnitTest( pCanvas );
	pUnit->SetPos( 10, 10 );

	//
	// Create a Windows Control helper 
	// (Processes Windows MSG's and fires input at GWEN)
	//
	//Gwen::Input::Windows GwenInput;
	//GwenInput.Initialize( pCanvas );

	//
	// Begin the main game loop
	//
//	MSG msg;
	while( !window->requestedExit() )
	{
		// Skip out if the window is closed
		//if ( !IsWindowVisible( g_pHWND ) )
			//break;

		// If we have a message from windows..
	//	if ( PeekMessage( &msg, NULL, 0, 0, PM_REMOVE ) )
		{

			// .. give it to the input handler to process
		//	GwenInput.ProcessMessage( msg );

			// if it's QUIT then quit..
		//	if ( msg.message == WM_QUIT )
			//	break;

			// Handle the regular window stuff..
		//	TranslateMessage(&msg);
		//	DispatchMessage(&msg);

		}

		window->startRendering();
		
		// Main OpenGL Render Loop
		{
			glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

				glEnable(GL_BLEND);
				GLint err = glGetError();
				assert(err==GL_NO_ERROR);

				glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		
				err = glGetError();
				assert(err==GL_NO_ERROR);

				err = glGetError();
				assert(err==GL_NO_ERROR);
        
				glDisable(GL_DEPTH_TEST);
				err = glGetError();
				assert(err==GL_NO_ERROR);
        
				//glColor4ub(255,0,0,255);
		
				err = glGetError();
				assert(err==GL_NO_ERROR);
        
		
				err = glGetError();
				assert(err==GL_NO_ERROR);
				glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);

			//	saveOpenGLState(width,height);//m_glutScreenWidth,m_glutScreenHeight);
			
				err = glGetError();
				assert(err==GL_NO_ERROR);

			
				err = glGetError();
				assert(err==GL_NO_ERROR);

				glDisable(GL_CULL_FACE);

				glDisable(GL_DEPTH_TEST);
				err = glGetError();
				assert(err==GL_NO_ERROR);

				err = glGetError();
				assert(err==GL_NO_ERROR);
            
				glEnable(GL_BLEND);

            
				err = glGetError();
				assert(err==GL_NO_ERROR);
            
 

			pCanvas->RenderCanvas();

	//		SwapBuffers( GetDC( g_pHWND ) );
		}
		window->endRendering();

	}

	window->closeWindow();
	delete window;
	

}