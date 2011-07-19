
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "Gwen/Gwen.h"
#include "Gwen/Skins/Simple.h"

#include "UnitTest.h"
#include "Gwen/Input/Windows.h"

#include "Gwen/Renderers/OpenGL_DebugFont.h"

#include "gl/glew.h"
HWND						g_pHWND = NULL;
Gwen::Controls::Canvas*		pCanvas  = NULL;
int sWidth = 1000;
int sHeight = 500;

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



int main()
{

	//
	// Create a new window
	//
	g_pHWND = CreateGameWindow();

	//
	// Create OpenGL Device
	//
	HGLRC OpenGLContext = CreateOpenGLDeviceContext();	


	//
	// Create a GWEN OpenGL Renderer
	//
		Gwen::Renderer::OpenGL_DebugFont * pRenderer = new Gwen::Renderer::OpenGL_DebugFont();

	//
	// Create a GWEN skin
	//
		 

#ifdef USE_TEXTURED_SKIN
	Gwen::Skin::TexturedBase skin;
	skin.SetRender( pRenderer );
	skin.Init("DefaultSkin.png");
#else
	Gwen::Skin::Simple skin;
	skin.SetRender( pRenderer );
#endif


	//
	// Create a Canvas (it's root, on which all other GWEN panels are created)
	//
	pCanvas = new Gwen::Controls::Canvas( &skin );
	pCanvas->SetSize( sWidth, sHeight);
	pCanvas->SetDrawBackground( true );
	pCanvas->SetBackgroundColor( Gwen::Color( 150, 170, 170, 255 ) );

	//
	// Create our unittest control (which is a Window with controls in it)
	//
	UnitTest* pUnit = new UnitTest( pCanvas );
	pUnit->SetPos( 10, 10 );

	//
	// Create a Windows Control helper 
	// (Processes Windows MSG's and fires input at GWEN)
	//
	Gwen::Input::Windows GwenInput;
	GwenInput.Initialize( pCanvas );

	//
	// Begin the main game loop
	//
	MSG msg;
	while( true )
	{
		// Skip out if the window is closed
		if ( !IsWindowVisible( g_pHWND ) )
			break;

		// If we have a message from windows..
		if ( PeekMessage( &msg, NULL, 0, 0, PM_REMOVE ) )
		{

			// .. give it to the input handler to process
			GwenInput.ProcessMessage( msg );

			// if it's QUIT then quit..
			if ( msg.message == WM_QUIT )
				break;

			// Handle the regular window stuff..
			TranslateMessage(&msg);
			DispatchMessage(&msg);

		}

		// Main OpenGL Render Loop
		{
			glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

			pCanvas->RenderCanvas();

			SwapBuffers( GetDC( g_pHWND ) );
		}
	}

	// Clean up OpenGL
	wglMakeCurrent( NULL, NULL );
	wglDeleteContext( OpenGLContext );

}