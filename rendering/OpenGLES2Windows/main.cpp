#include <iostream>
#include <stdlib.h> 
#include <ctime>
#include <windows.h>
#include <math.h>

#include "GLES2/gl2.h"
#include "EGL/egl.h"

#include "sbm.h"

#pragma comment(lib, "libEGL.lib")
#pragma comment(lib, "libGLESv2.lib")
bool opengl_initialized = false;

class RenderState
{
public:
    RenderState() : po(0), vertLoc(0), fragColorLoc(0), mvpLoc(0), normalLoc(0), 
        texcoordLoc(0), texUnitLoc(0)
    {}
    ~RenderState() {}

    GLint po;
    GLint vertLoc;
    GLint fragColorLoc;
    GLint mvpLoc;
    GLint normalLoc;
    GLint texcoordLoc;
    GLint texUnitLoc;

    SBObject            ninja;
    GLuint              ninjaTex[1];

};

class esContext 
{
public:
    esContext() : 
        ESdll(0), eglDisplay(0), eglSurface(0), eglContext(0), 
        hInstance(0), hWnd(0), hDC(0), nWindowWidth(0),
        nWindowHeight(0), nWindowX(0), nWindowY(0)
    {}

    ~esContext() {}

    HINSTANCE ESdll;

    EGLDisplay eglDisplay;
    EGLSurface eglSurface;
    EGLContext eglContext;

    HINSTANCE   hInstance ;
    HWND        hWnd;
    HDC         hDC;
    WNDCLASS    windClass; 
    RECT        windowRect;

    int         nWindowWidth;
    int         nWindowHeight;
    int         nWindowX;
    int         nWindowY;

    RenderState rs;
};

GLfloat vWhite[] = { 1.0, 1.0, 1.0, 1.0 };

GLfloat mvpMatrix[] = {
    (float)0.047573917,
    (float)0.0,
    (float)0.0,
    (float)0.0,
    (float)0.0,
    (float)0.019029567,
    (float)0.0,
    (float)0.0,
    (float)0.0,
    (float)0.0,
    (float)-0.020404041,
    (float)-0.02,
    (float)0.0,
    (float)-1.2686380,
    (float)0.53030318,
    (float)2.5  };

#pragma pack(1)
struct RGB { 
  GLbyte blue;
  GLbyte green;
  GLbyte red;
  GLbyte alpha;
};

struct BMPInfoHeader {
  GLuint	size;
  GLuint	width;
  GLuint	height;
  GLushort  planes;
  GLushort  bits;
  GLuint	compression;
  GLuint	imageSize;
  GLuint	xScale;
  GLuint	yScale;
  GLuint	colors;
  GLuint	importantColors;
};

struct BMPHeader {
  GLushort	type; 
  GLuint	size; 
  GLushort	unused; 
  GLushort	unused2; 
  GLuint	offset; 
}; 

struct BMPInfo {
  BMPInfoHeader		header;
  RGB				colors[1];
};

#pragma pack(8)

esContext ctx;

using namespace std;

void ResizeViewport(int nWidth, int nHeight)
{
	if (opengl_initialized)
		glViewport(0, 0, nWidth, nHeight);
}

LRESULT CALLBACK WndProc(	HWND	hWnd,  UINT	uMsg,  WPARAM	wParam,	 LPARAM	lParam)
{
    unsigned int key = 0;
    // Handle relevant messages individually
    switch(uMsg)
    {
    case WM_ACTIVATE:
    case WM_SETFOCUS:
        return 0;
    case WM_SIZE:
        ResizeViewport(LOWORD(lParam), HIWORD(lParam));
        break;
    case WM_CLOSE:
        DestroyWindow(hWnd); 
        return 0; 
    case WM_DESTROY: 
        eglMakeCurrent(EGL_NO_DISPLAY, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        eglDestroyContext(ctx.eglDisplay, ctx.eglContext);
        eglDestroySurface(ctx.eglDisplay, ctx.eglSurface);
        eglTerminate(ctx.eglDisplay);
        PostQuitMessage(0);
        return 0;
    case WM_KEYDOWN:
        switch (wParam)
        {
			case VK_ESCAPE:
	            PostMessage(hWnd, WM_CLOSE, 0, 0);
                return 0;
            default:
                break;
        }
        break;
    default:
        break;
    }

    return DefWindowProc(hWnd,uMsg,wParam,lParam);
}

bool CreateNewWindow(esContext &ctx)
{
    bool bRetVal = true;

    DWORD dwExtStyle;
    DWORD dwWindStyle;

    ctx.hInstance = GetModuleHandle(NULL);

    TCHAR szWindowName[50] =  TEXT("OpenGL ES Sample");
    TCHAR szClassName[50]  =  TEXT("OGL_CLASS");

    // setup window class
    ctx.windClass.lpszClassName = szClassName;                // Set the name of the Class
    ctx.windClass.lpfnWndProc   = (WNDPROC)WndProc;
    ctx.windClass.hInstance     = ctx.hInstance;                // Use this module for the module handle
    ctx.windClass.hCursor       = LoadCursor(NULL, IDC_ARROW);// Pick the default mouse cursor
    ctx.windClass.hIcon         = LoadIcon(NULL, IDI_WINLOGO);// Pick the default windows icons
    ctx.windClass.hbrBackground = NULL;                       // No Background
    ctx.windClass.lpszMenuName  = NULL;                       // No menu for this window
    ctx.windClass.style         = CS_HREDRAW | CS_OWNDC |     // set styles for this class, specifically to catch
                                    CS_VREDRAW;                 // window redraws, unique DC, and resize
    ctx.windClass.cbClsExtra    = 0;                          // Extra class memory
    ctx.windClass.cbWndExtra    = 0;                          // Extra window memory

    // Register the newly defined class
    if(!RegisterClass( &ctx.windClass ))
        bRetVal = false;

    dwExtStyle  = WS_EX_APPWINDOW | WS_EX_WINDOWEDGE;
    dwWindStyle = WS_OVERLAPPEDWINDOW;
    ShowCursor(TRUE);

    ctx.windowRect.left   = ctx.nWindowX;
    ctx.windowRect.right  = ctx.nWindowX + ctx.nWindowWidth;
    ctx.windowRect.top    = ctx.nWindowY;
    ctx.windowRect.bottom = ctx.nWindowY + ctx.nWindowHeight;

    // Setup window width and height
    AdjustWindowRectEx(&ctx.windowRect, dwWindStyle, FALSE, dwExtStyle);

    //Adjust for adornments
    int nWindowWidth  = ctx.windowRect.right  - ctx.windowRect.left;
    int nWindowHeight = ctx.windowRect.bottom - ctx.windowRect.top;

    // Create window
    ctx.hWnd = CreateWindowEx(dwExtStyle,     // Extended style
                              szClassName,    // class name
                              szWindowName,   // window name
                              dwWindStyle |        
                              WS_CLIPSIBLINGS | 
                              WS_CLIPCHILDREN,// window stlye
                              ctx.nWindowX,   // window position, x
                              ctx.nWindowY,   // window position, y
                              nWindowWidth,   // height
                              nWindowHeight,  // width
                              NULL,           // Parent window
                              NULL,           // menu
                              ctx.hInstance,    // instance
                              NULL);          // pass this to WM_CREATE

    // now that we have a window, setup the pixel format descriptor
    ctx.hDC = GetDC(ctx.hWnd);
    
    ShowWindow(ctx.hWnd, SW_SHOWDEFAULT);

    return bRetVal;
}

GLint CreateShader(GLenum type, const char *str)
{
    GLint status;
    GLint name = glCreateShader(type);
    if(name == 0)
    return 0;

    glShaderSource(name, 1, &str, NULL );
    glCompileShader(name);
    glGetShaderiv(name, GL_COMPILE_STATUS, &status );

    if(status == 0) 
    {
        glDeleteShader(name);
        return 0;
    }

    return name;
}

GLint CreateProgram ( const char *vertShaderSrc, const char *fragShaderSrc )
{
    GLuint vs, fs, po;
    GLint status;

    // Load the vertex/fragment shaders
    vs = CreateShader(GL_VERTEX_SHADER, vertShaderSrc);
    if ( vs == 0 )
    {
        printf("Failed to create a vertex shader.\n");
        return 0;
    }

    fs = CreateShader(GL_FRAGMENT_SHADER, fragShaderSrc);
    if (fs == 0)
    {
        printf("Failed to create a fragment shader.\n");
        glDeleteShader(vs);
        return 0;
    }

    po = glCreateProgram ( );

    if (po == 0)
    {
        printf("Failed to create a program.\n");
        return 0;
    }

    glAttachShader(po, vs);
    glAttachShader(po, fs);

    glBindAttribLocation(po, 0, "vertPosition");
    glBindAttribLocation(po, 1, "vNormal");
    glBindAttribLocation(po, 2, "texCoord0");
    glLinkProgram(po);

    glGetProgramiv(po, GL_LINK_STATUS, &status);

    if(!status) 
    {
        printf("Failed to link program.\n");
        glDeleteProgram(po);
        return 0;
    }

    // Free up no longer needed shader resources
    glDeleteShader(vs);
    glDeleteShader(fs);

    return po;
}

bool LoadTexture(esContext &  tx)
{
    FILE*	pFile;
	BMPInfo *pBitmapInfo = NULL;
	unsigned long lInfoSize = 0;
	unsigned long lBitSize = 0;
	GLbyte *pBits = NULL;					
	BMPHeader	bitmapHeader;

    fopen_s(&pFile, "Ninja/NinjaComp.bmp", "rb");
    if(pFile == NULL)
        return false;

    fread(&bitmapHeader, sizeof(BMPHeader), 1, pFile);

	lInfoSize = bitmapHeader.offset - sizeof(BMPHeader);
	pBitmapInfo = (BMPInfo *) malloc(sizeof(GLbyte)*lInfoSize);
	if(fread(pBitmapInfo, lInfoSize, 1, pFile) != 1)
	{
		free(pBitmapInfo);
		fclose(pFile);
        printf("Failed to load texture.\n");
		return false;
	}

	GLint nWidth = pBitmapInfo->header.width;
	GLint nHeight = pBitmapInfo->header.height;
	lBitSize = pBitmapInfo->header.imageSize;

	if(pBitmapInfo->header.bits != 24)
	{
        printf("Failed to load texture.\n");
		free(pBitmapInfo);
		return false;
	}

	if(lBitSize == 0)
		lBitSize = (nWidth *
           pBitmapInfo->header.bits + 7) / 8 *
  		  abs(nHeight);

	free(pBitmapInfo);
	pBits = (GLbyte*)malloc(sizeof(GLbyte)*lBitSize);

	if(fread(pBits, lBitSize, 1, pFile) != 1)
	{
		free(pBits);
		pBits = NULL;
	}

	fclose(pFile);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, nWidth, nHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, pBits);

    return true;
}

bool InitState(esContext &ctx)
{
    char vertShaderStr[] =  
      "attribute vec4 vertPosition;   \n"
      "attribute vec3 vNormal;        \n"
      "attribute vec2 texCoord0;      \n"
      "uniform mat4 mvpMatrix;        \n"
      "varying vec2 vTexCoord;        \n"
      "void main()                    \n"
      "{                              \n"
      "   gl_Position = mvpMatrix * vertPosition; \n"
      "   vTexCoord   = texCoord0;    \n"
      "}                              \n";

    char fragShaderStr[] =  
      "precision highp float;          \n"
      "varying vec2 vTexCoord;         \n"
      "uniform sampler2D textureUnit0; \n" 
      "void main()                     \n"
      "{                               \n"
      "  gl_FragColor =  texture(textureUnit0, vTexCoord); \n"
      "}                               \n";

    ctx.rs.po = CreateProgram((char*)vertShaderStr, (char*)fragShaderStr);

    if (ctx.rs.po == 0)
        return false;

    ctx.rs.vertLoc      = glGetAttribLocation( ctx.rs.po, "vertPosition" );
    ctx.rs.normalLoc    = glGetAttribLocation( ctx.rs.po, "vNormal" );
    ctx.rs.texcoordLoc  = glGetAttribLocation( ctx.rs.po, "texCoord0" );
    ctx.rs.fragColorLoc = glGetUniformLocation( ctx.rs.po, "fragColor" );
    ctx.rs.mvpLoc       = glGetUniformLocation( ctx.rs.po, "mvpMatrix" );
    ctx.rs.texUnitLoc   = glGetUniformLocation( ctx.rs.po, "textureUnit0" );

    glClearColor ( 0.7f, 0.7f, 0.7f, 0.0f );

    if (!ctx.rs.ninja.LoadFromSBM("./Ninja/ninja.sbm"))
        return false;

    glActiveTexture(GL_TEXTURE0);
    glGenTextures(1, ctx.rs.ninjaTex);
    glBindTexture(GL_TEXTURE_2D, ctx.rs.ninjaTex[0]);
    if (!LoadTexture(ctx))
        return false;

    return true;
}

void SetupProgram(esContext &ctx)
{
    glUseProgram(ctx.rs.po);
    glUniformMatrix4fv(ctx.rs.mvpLoc, 1, GL_FALSE, mvpMatrix);

    glUniform4fv(ctx.rs.fragColorLoc, 1, vWhite);

    GLfloat texLoc[] = { 0.0f };
    glUniform4fv(ctx.rs.texUnitLoc,   1, texLoc);
}

void RenderScene(esContext &ctx)
{
    glClearColor(1.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    SetupProgram(ctx);

    ctx.rs.ninja.Render();
    eglSwapBuffers(ctx.eglDisplay, ctx.eglSurface);
}

int main(int argc, char** argv) {

    ctx.nWindowWidth  = 640;
    ctx.nWindowHeight = 480;

    CreateNewWindow(ctx);

    int lRet = 0;
	//EGLint attrs[] = { EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE, EGL_NONE };

EGLint attrs[] =
   {
       EGL_RED_SIZE,       5,
       EGL_GREEN_SIZE,     6,
       EGL_BLUE_SIZE,      5,
       EGL_ALPHA_SIZE,     EGL_DONT_CARE,
       EGL_DEPTH_SIZE,     EGL_DONT_CARE,
       EGL_STENCIL_SIZE,   EGL_DONT_CARE,
       EGL_SAMPLE_BUFFERS, 0,
       EGL_NONE
   };

    //EGLint attrs[3] = { EGL_DEPTH_SIZE, 16, EGL_NONE };
    EGLint numConfig =0;
    EGLConfig eglConfig = 0;

    if ((ctx.eglDisplay = eglGetDisplay((NativeDisplayType)NULL)) == EGL_NO_DISPLAY)
    { PostQuitMessage(0); return lRet; }

    // Initialize the display
    EGLint major = 0;
    EGLint minor = 0;
    if (!eglInitialize(ctx.eglDisplay, &major, &minor))
    { PostQuitMessage(0); return lRet; }

    if (major < 1 ||
        minor < 4)
    {
        // Does not support EGL 1.4
        lRet = 1;
        printf("System does not support at least EGL 1.4 \n");
        return lRet;
    }

	if ( !eglGetConfigs(ctx.eglDisplay, NULL, 0, &numConfig) )
	{
		PostQuitMessage(0); return lRet; 
	}
    // Obtain the first configuration with a depth buffer
    if (!eglChooseConfig(ctx.eglDisplay, attrs, &eglConfig, 1, &numConfig))
    { PostQuitMessage(0); return lRet; }

	  ctx.eglSurface = eglCreateWindowSurface(ctx.eglDisplay, eglConfig, (EGLNativeWindowType)ctx.hWnd, NULL);
    // Create a surface for the main window
    
	  if (ctx.eglSurface==EGL_NO_SURFACE)
    { PostQuitMessage(0); return lRet; }

	   EGLint contextAttribs[] = { EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE, EGL_NONE };

    // Create an OpenGL ES context
    ctx.eglContext = eglCreateContext(ctx.eglDisplay, eglConfig, EGL_NO_CONTEXT, NULL);
	if (ctx.eglContext == EGL_NO_CONTEXT)
    { PostQuitMessage(0); return lRet; }

    // Make the context and surface current
    if (!eglMakeCurrent(ctx.eglDisplay, ctx.eglSurface, ctx.eglSurface, ctx.eglContext))
    { PostQuitMessage(0); return lRet; }
    
    if (!InitState(ctx))
    {
        printf("Failed to Setup state.\n");
        exit(0);
    }

	opengl_initialized = true;

    ResizeViewport(ctx.nWindowWidth, ctx.nWindowHeight);

    int done = 0;
    while (!done)
    {
        RenderScene(ctx);
        // Peek or wait for messages
        MSG msg;
        while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
        {
            if (msg.message==WM_QUIT)
            {
                done = 1; 
            }
            else
            {
                TranslateMessage(&msg); 
                DispatchMessage(&msg); 
            }
        }
    }

    return lRet; 
}



SBObject::SBObject(void)
    : m_vao(0),
      m_attribute_buffer(0),
      m_index_buffer(0),
      m_attrib(0),
      m_frame(0)
{

}

SBObject::~SBObject(void)
{
    Free();
}

bool SBObject::LoadFromSBM(const char * filename)
{
    FILE * f = NULL;

    fopen_s(&f, filename, "rb");

    fseek(f, 0, SEEK_END);
    size_t filesize = ftell(f);
    fseek(f, 0, SEEK_SET);

    m_data = new unsigned char [filesize];
    fread(m_data, filesize, 1, f);
    fclose(f);

    SBM_HEADER * header = (SBM_HEADER *)m_data;
    m_raw_data = m_data + sizeof(SBM_HEADER) + header->num_attribs * sizeof(SBM_ATTRIB_HEADER) + header->num_frames * sizeof(SBM_FRAME_HEADER);
    SBM_ATTRIB_HEADER * attrib_header = (SBM_ATTRIB_HEADER *)(m_data + sizeof(SBM_HEADER));
    SBM_FRAME_HEADER * frame_header = (SBM_FRAME_HEADER *)(m_data + sizeof(SBM_HEADER) + header->num_attribs * sizeof(SBM_ATTRIB_HEADER));

    memcpy(&m_header, header, sizeof(SBM_HEADER));
    m_attrib = new SBM_ATTRIB_HEADER[header->num_attribs];
    memcpy(m_attrib, attrib_header, header->num_attribs * sizeof(SBM_ATTRIB_HEADER));
    m_frame = new SBM_FRAME_HEADER[header->num_frames];
    memcpy(m_frame, frame_header, header->num_frames * sizeof(SBM_FRAME_HEADER));

    return true;
}

bool SBObject::Free(void)
{
    m_index_buffer = 0;
    m_attribute_buffer = 0;
    m_vao = 0;

    delete [] m_attrib;
    m_attrib = NULL;

    delete [] m_frame;
    m_frame = NULL;
    
    delete [] m_data;

    return true;
}

void SBObject::Render()
{
    unsigned char * data_pointer = m_raw_data;
        
    glVertexAttribPointer(0, m_attrib[0].components, m_attrib[0].type, GL_FALSE, 0, (GLvoid *)data_pointer);
    glEnableVertexAttribArray(0);
    data_pointer += m_attrib[0].components * sizeof(GLfloat) * m_header.num_vertices;

    glVertexAttribPointer(1, m_attrib[1].components, m_attrib[1].type, GL_FALSE, 0, (GLvoid *)data_pointer);
    glEnableVertexAttribArray(1);
    data_pointer += m_attrib[1].components * sizeof(GLfloat) * m_header.num_vertices;

    glVertexAttribPointer(2, m_attrib[2].components, m_attrib[2].type, GL_FALSE, 0, (GLvoid *)data_pointer);
    glEnableVertexAttribArray(2);
    data_pointer += m_attrib[2].components * sizeof(GLfloat) * m_header.num_vertices;

    glDrawArrays(GL_TRIANGLES, m_frame[0].first, m_frame[0].count);
}

