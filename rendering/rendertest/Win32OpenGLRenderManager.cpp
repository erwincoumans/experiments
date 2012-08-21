/*
Copyright (c) 2012 Advanced Micro Devices, Inc.  

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/
//Originally written by Erwin Coumans


#include "Win32OpenGLRenderManager.h"

#include <windows.h>
#include <GL/gl.h>
#include "LinearMath/btVector3.h"

static InternalData2* sData = 0;



extern bool printStats;
extern bool pauseSimulation;
extern bool shootObject;
extern int m_glutScreenWidth;
extern int m_glutScreenHeight;

struct InternalData2
{
	HWND m_hWnd;;
	int m_fullWindowWidth;//includes borders etc
	int m_fullWindowHeight;

	int m_openglViewportWidth;//just the 3d viewport/client area
	int m_openglViewportHeight;

	HDC m_hDC;
	HGLRC m_hRC;
	bool m_OpenGLInitialized;
	int m_oldScreenWidth;
	int m_oldHeight;
	int m_oldBitsPerPel;
	bool m_quit;
	Win32OpenGLWindow* m_window;
	int m_mouseLButton;
	int m_mouseRButton;
	int m_mouseMButton;
	int m_mouseXpos;
	int m_mouseYpos;

	btWheelCallback m_wheelCallback;
	btMouseCallback	m_mouseCallback;
	btKeyboardCallback	m_keyboardCallback;

	
	
	InternalData2()
	{
		m_hWnd = 0;
		m_window = 0;
		m_mouseLButton=0;
		m_mouseRButton=0;
		m_mouseMButton=0;
	
		m_fullWindowWidth = 0;
		m_fullWindowHeight= 0;
		m_openglViewportHeight=0;
		m_openglViewportWidth=0;
		m_hDC = 0;
		m_hRC = 0;
		m_OpenGLInitialized = false;
		m_oldScreenWidth = 0;
		m_oldHeight = 0;
		m_oldBitsPerPel = 0;
		m_quit = false;

		m_keyboardCallback = 0;
		m_mouseCallback = 0;
		m_wheelCallback = 0;
	}
};


void Win32OpenGLWindow::enableOpenGL()
{
	
	
	
	PIXELFORMATDESCRIPTOR pfd;
	int format;
	
	// get the device context (DC)
	m_data->m_hDC = GetDC( m_data->m_hWnd );
	
	// set the pixel format for the DC
	ZeroMemory( &pfd, sizeof( pfd ) );
	pfd.nSize = sizeof( pfd );
	pfd.nVersion = 1;
	pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
	pfd.iPixelType = PFD_TYPE_RGBA;
	pfd.cColorBits = 32;
	pfd.cDepthBits = 16;
	pfd.cStencilBits = 1;
	pfd.iLayerType = PFD_MAIN_PLANE;
	format = ChoosePixelFormat( m_data->m_hDC, &pfd );
	SetPixelFormat( m_data->m_hDC, format, &pfd );
	
	// create and enable the render context (RC)
	m_data->m_hRC = wglCreateContext( m_data->m_hDC );
	wglMakeCurrent( m_data->m_hDC, m_data->m_hRC );
	m_data->m_OpenGLInitialized = true;
	
	
}

void Win32OpenGLWindow::getMouseCoordinates(int& x, int& y)
{
	x = m_data->m_mouseXpos;
	y = m_data->m_mouseYpos;
}


void Win32OpenGLWindow::disableOpenGL()
{
	m_data->m_OpenGLInitialized = false;

	wglMakeCurrent( NULL, NULL );
	wglDeleteContext( m_data->m_hRC );
	ReleaseDC( m_data->m_hWnd, m_data->m_hDC );
}

void Win32OpenGLWindow::pumpMessage()
{
	MSG msg;
		// check for messages
		while( PeekMessage( &msg, NULL, 0, 0, PM_REMOVE )  )
		{
			
			// handle or dispatch messages
			if ( msg.message == WM_QUIT ) 
			{
				m_data->m_quit = TRUE;
			} 
			else 
			{
				TranslateMessage( &msg );
				DispatchMessage( &msg );
			}
			
//			gDemoApplication->displayCallback();
			

		};
}


LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	switch (message)
	{
	case WM_PAINT:
		{
			PAINTSTRUCT ps;
			BeginPaint(hWnd, &ps);
			EndPaint(hWnd, &ps);
		}
		return 0;

	case WM_ERASEBKGND:
		return 0;
	
	case WM_CLOSE:
		PostQuitMessage(0);
		return 0;

	case WM_DESTROY:
		PostQuitMessage(0);
		return 0;

	case WM_KEYDOWN:
		{
			switch ( wParam )
			{
			case ' ':
			case 'P':
			case 'p':
				{
					pauseSimulation = !pauseSimulation;
					break;
				}
			case 'S':
			case 's':
				{
					printStats=true;
					break;
				}
				case 'Q':
				case VK_ESCAPE:
					{
						PostQuitMessage(0);
					}
					return 0;
			}
			break;
		}

		case WM_MBUTTONUP:
	{
			int xPos = LOWORD(lParam); 
			int yPos = HIWORD(lParam); 
			if (sData)
			{
				sData->m_mouseMButton=0;
				sData->m_mouseXpos = xPos;
				sData->m_mouseYpos = yPos;
			}
		break;
	}
	case WM_MBUTTONDOWN:
	{
			int xPos = LOWORD(lParam); 
			int yPos = HIWORD(lParam); 
			if (sData)
			{
				sData->m_mouseMButton=1;
				sData->m_mouseXpos = xPos;
				sData->m_mouseYpos = yPos;
			}
		break;
	}

	case WM_LBUTTONUP:
	{
			int xPos = LOWORD(lParam); 
			int yPos = HIWORD(lParam); 
			if (sData)
			{
				sData->m_mouseLButton=0;
				sData->m_mouseXpos = xPos;
				sData->m_mouseYpos = yPos;
				
				if (sData && sData->m_mouseCallback)
					(*sData->m_mouseCallback)(0,0,xPos,yPos);

			}
		//	gDemoApplication->mouseFunc(0,1,xPos,yPos);
		break;
	}
	case WM_LBUTTONDOWN:
		{
				int xPos = LOWORD(lParam); 
				int yPos = HIWORD(lParam); 
			if (sData)
			{
				sData->m_mouseLButton=1;
				sData->m_mouseXpos = xPos;
				sData->m_mouseYpos = yPos;

				if (sData && sData->m_mouseCallback)
					(*sData->m_mouseCallback)(0,1,xPos,yPos);
			}
			break;
		}


	case 0x020A://WM_MOUSEWHEEL:
	{

		int  zDelta = (short)HIWORD(wParam);
		int xPos = LOWORD(lParam); 
		int yPos = HIWORD(lParam); 
		//m_cameraDistance -= zDelta*0.01;
		if (sData && sData->m_wheelCallback)
			(*sData->m_wheelCallback)(0,float(zDelta)*0.05f);

		break;
	}

	case WM_MOUSEMOVE:
		{
				int xPos = LOWORD(lParam); 
				int yPos = HIWORD(lParam); 
				sData->m_mouseXpos = xPos;
				sData->m_mouseYpos = yPos;

				if (sData && sData->m_mouseCallback)
					(*sData->m_mouseCallback)(-1,0,xPos,yPos);

			break;
		}
	case WM_RBUTTONUP:
	{
			int xPos = LOWORD(lParam); 
			int yPos = HIWORD(lParam); 
			sData->m_mouseRButton = 1;

			//gDemoApplication->mouseFunc(2,1,xPos,yPos);
		break;
	}
	case WM_RBUTTONDOWN:
	{
			int xPos = LOWORD(lParam); 
			int yPos = HIWORD(lParam); 
			sData->m_mouseRButton = 0;
			shootObject = true;

			//gDemoApplication->mouseFunc(2,0,xPos,yPos);
		break;
	}
	case WM_SIZE:													// Size Action Has Taken Place

			RECT clientRect;
			GetClientRect(hWnd,&clientRect);

			switch (wParam)												// Evaluate Size Action
			{

				case SIZE_MINIMIZED:									// Was Window Minimized?
				return 0;												// Return

				case SIZE_MAXIMIZED:									// Was Window Maximized?
				case SIZE_RESTORED:										// Was Window Restored?
					RECT wr;
					GetWindowRect(hWnd,&wr);
					
					sData->m_fullWindowWidth = wr.right-wr.left;
					sData->m_fullWindowHeight = wr.bottom-wr.top;//LOWORD (lParam) HIWORD (lParam);
					sData->m_openglViewportWidth = clientRect.right;
					sData->m_openglViewportHeight = clientRect.bottom;
					m_glutScreenWidth = sData->m_openglViewportWidth;
					m_glutScreenHeight = sData->m_openglViewportHeight;
					//if (sOpenGLInitialized)
					//{
					//	//gDemoApplication->reshape(sWidth,sHeight);
					//}
					glViewport(0, 0, sData->m_openglViewportWidth, sData->m_openglViewportHeight);
				return 0;												// Return
			}
		break;

	default:{

			}
	};

	return DefWindowProc(hWnd, message, wParam, lParam);
}




void	Win32OpenGLWindow::init(int oglViewportWidth,int oglViewportHeight, bool fullscreen,int colorBitsPerPixel, void* windowHandle)
{
	// get handle to exe file
	HINSTANCE hInstance = GetModuleHandle(0);


	// create the window if we need to and we do not use the null device
	if (!windowHandle)
	{
		const char* ClassName = "DeviceWin32";

		// Register Class
		WNDCLASSEX wcex;
		wcex.cbSize		= sizeof(WNDCLASSEX);
		wcex.style		= CS_HREDRAW | CS_VREDRAW;
		wcex.lpfnWndProc	= WndProc;
		wcex.cbClsExtra		= 0;
		wcex.cbWndExtra		= 0;
		wcex.hInstance		= hInstance;
		wcex.hIcon		= LoadIcon( NULL, IDI_APPLICATION ); //(HICON)LoadImage(hInstance, "bullet_ico.ico", IMAGE_ICON, 0,0, LR_LOADTRANSPARENT);//LR_LOADFROMFILE);
		wcex.hCursor		= LoadCursor(NULL, IDC_ARROW);
		wcex.hbrBackground	= (HBRUSH)(COLOR_WINDOW+1);
		wcex.lpszMenuName	= 0;
		wcex.lpszClassName	= ClassName;
		wcex.hIconSm		= 0;

		// if there is an icon, load it
//		wcex.hIcon = (HICON)LoadImage(hInstance, "bullet.ico", IMAGE_ICON, 0,0, LR_LOADFROMFILE);

		RegisterClassEx(&wcex);

		// calculate client size

		RECT clientSize;
		clientSize.top = 0;
		clientSize.left = 0;
		clientSize.right = oglViewportWidth;
		clientSize.bottom = oglViewportHeight;

		DWORD style = WS_POPUP;

		if (!fullscreen)
			style = WS_SYSMENU | WS_BORDER | WS_CAPTION | WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_MINIMIZEBOX | WS_MAXIMIZEBOX | WS_SIZEBOX;

		AdjustWindowRect(&clientSize, style, false);
		
		m_data->m_fullWindowWidth = clientSize.right - clientSize.left;
		m_data->m_fullWindowHeight = clientSize.bottom - clientSize.top;

		int windowLeft = (GetSystemMetrics(SM_CXSCREEN) - m_data->m_fullWindowWidth) / 2;
		int windowTop = (GetSystemMetrics(SM_CYSCREEN) - m_data->m_fullWindowHeight) / 2;

		if (fullscreen)
		{
			windowLeft = 0;
			windowTop = 0;
		}

		// create window

		m_data->m_hWnd = CreateWindow( ClassName, "", style, windowLeft, windowTop,
			m_data->m_fullWindowWidth, m_data->m_fullWindowHeight,NULL, NULL, hInstance, NULL);

		
		RECT clientRect;
		GetClientRect(m_data->m_hWnd,&clientRect);



		ShowWindow(m_data->m_hWnd, SW_SHOW);
		UpdateWindow(m_data->m_hWnd);

		MoveWindow(m_data->m_hWnd, windowLeft, windowTop, m_data->m_fullWindowWidth, m_data->m_fullWindowHeight, TRUE);

		GetClientRect(m_data->m_hWnd,&clientRect);
		int w = clientRect.right-clientRect.left;
		int h = clientRect.bottom-clientRect.top;
//		printf("actual client OpenGL viewport width / height = %d, %d\n",w,h);
		
	}
	else if (windowHandle)
	{
		// attach external window
		m_data->m_hWnd = static_cast<HWND>(windowHandle);
		RECT r;
		GetWindowRect(m_data->m_hWnd, &r);
		m_data->m_fullWindowWidth = r.right - r.left;
		m_data->m_fullWindowHeight= r.bottom - r.top;

		m_glutScreenWidth = sData->m_openglViewportWidth;
		m_glutScreenHeight = sData->m_openglViewportHeight;

		//sFullScreen = false;
		//sExternalWindow = true;
	}


	if (fullscreen)
	{
		DEVMODE dm;
		memset(&dm, 0, sizeof(dm));
		dm.dmSize = sizeof(dm);
		// use default values from current setting
		EnumDisplaySettings(NULL, ENUM_CURRENT_SETTINGS, &dm);
		m_data->m_oldScreenWidth = dm.dmPelsWidth;
		m_data->m_oldHeight = dm.dmPelsHeight;
		m_data->m_oldBitsPerPel = dm.dmBitsPerPel;

		dm.dmPelsWidth = oglViewportWidth;
		dm.dmPelsHeight = oglViewportHeight;
		if (colorBitsPerPixel)
		{
			dm.dmBitsPerPel = colorBitsPerPixel;
		}
		dm.dmFields = DM_BITSPERPEL | DM_PELSWIDTH | DM_PELSHEIGHT | DM_DISPLAYFREQUENCY;

		LONG res = ChangeDisplaySettings(&dm, CDS_FULLSCREEN);
		if (res != DISP_CHANGE_SUCCESSFUL)
		{ // try again without forcing display frequency
			dm.dmFields = DM_BITSPERPEL | DM_PELSWIDTH | DM_PELSHEIGHT;
			res = ChangeDisplaySettings(&dm, CDS_FULLSCREEN);
		}

	}

	//VideoDriver = video::createOpenGLDriver(CreationParams, FileSystem, this);
	enableOpenGL();


	const wchar_t* text= L"OpenCL rigid body demo";

	DWORD dwResult;

#ifdef _WIN64
		SetWindowTextW(m_data->m_hWnd, text);
#else
		SendMessageTimeoutW(m_data->m_hWnd, WM_SETTEXT, 0,
				reinterpret_cast<LPARAM>(text),
				SMTO_ABORTIFHUNG, 2000, &dwResult);
#endif
	

}


void	Win32OpenGLWindow::switchFullScreen(bool fullscreen,int width,int height,int colorBitsPerPixel)
{
	LONG res;
	DEVMODE dm;
	memset(&dm, 0, sizeof(dm));
	dm.dmSize = sizeof(dm);
	// use default values from current setting
	EnumDisplaySettings(NULL, ENUM_CURRENT_SETTINGS, &dm);

	dm.dmFields = DM_BITSPERPEL | DM_PELSWIDTH | DM_PELSHEIGHT | DM_DISPLAYFREQUENCY;

	if (fullscreen && !m_data->m_oldScreenWidth)
	{
		m_data->m_oldScreenWidth = dm.dmPelsWidth;
		m_data->m_oldHeight = dm.dmPelsHeight;
		m_data->m_oldBitsPerPel = dm.dmBitsPerPel;

		if (width && height)
		{
			dm.dmPelsWidth = width;
			dm.dmPelsHeight = height;
		} else
		{
			dm.dmPelsWidth = m_data->m_fullWindowWidth;
			dm.dmPelsHeight = m_data->m_fullWindowHeight;
		}
		if (colorBitsPerPixel)
		{
			dm.dmBitsPerPel = colorBitsPerPixel;
		}
	} else
	{
		if (m_data->m_oldScreenWidth)
		{
			dm.dmPelsWidth =	m_data->m_oldScreenWidth;
			dm.dmPelsHeight=	m_data->m_oldHeight;
			dm.dmBitsPerPel =   m_data->m_oldBitsPerPel;
		}
	}

	if (fullscreen)
	{
		res = ChangeDisplaySettings(&dm, CDS_FULLSCREEN);
	} else
	{
		res = ChangeDisplaySettings(&dm, 0);
	}
}



Win32OpenGLWindow::Win32OpenGLWindow()
{
	m_data = new InternalData2();
	sData = m_data;
	
}

Win32OpenGLWindow::~Win32OpenGLWindow()
{
	setKeyboardCallback(0);
	setMouseCallback(0);
	
	sData = 0;
	delete m_data;
	
}

void	Win32OpenGLWindow::init()
{
	init(640,480,false);
}


void	Win32OpenGLWindow::exit()
{
	setKeyboardCallback(0);
	setMouseCallback(0);

	
	
	disableOpenGL();
	DestroyWindow(this->m_data->m_hWnd);
}


void Win32OpenGLWindow::runMainLoop()
{

}


void	Win32OpenGLWindow::startRendering()
{
		pumpMessage();

		//glClearColor(1.f,0.f,0.f,1.f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);	//clear buffers
		
		//glCullFace(GL_BACK);
		//glFrontFace(GL_CCW);
		glEnable(GL_DEPTH_TEST);


		float aspect;

		if (m_data->m_openglViewportWidth > m_data->m_openglViewportHeight) 
		{
			aspect = (float)m_data->m_openglViewportWidth / (float)m_data->m_openglViewportHeight;
		} else 
		{
			aspect = (float)m_data->m_openglViewportHeight / (float)m_data->m_openglViewportWidth;
		}
	
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();

		if (m_data->m_openglViewportWidth> m_data->m_openglViewportHeight) 
		{
			glFrustum (-aspect, aspect, -1.0, 1.0, 1.0, 10000.0);
		} else 
		{
			glFrustum (-1.0, 1.0, -aspect, aspect, 1.0, 10000.0);
		}
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

}


void	Win32OpenGLWindow::renderAllObjects()
{
}

void	Win32OpenGLWindow::endRendering()
{
	SwapBuffers( m_data->m_hDC );
}

float	Win32OpenGLWindow::getTimeInSeconds()
{
	return 0.f;
}

void	Win32OpenGLWindow::setDebugMessage(int x,int y,const char* message)
{
}

bool Win32OpenGLWindow::requestedExit()
{
	return m_data->m_quit;
}

void Win32OpenGLWindow::setWheelCallback(btWheelCallback wheelCallback)
{
	m_data->m_wheelCallback = wheelCallback;
}

void Win32OpenGLWindow::setMouseCallback(btMouseCallback	mouseCallback)
{
	m_data->m_mouseCallback = mouseCallback;
	
}

void Win32OpenGLWindow::setKeyboardCallback( btKeyboardCallback	keyboardCallback)
{
	m_data->m_keyboardCallback = keyboardCallback;
	
}


	
