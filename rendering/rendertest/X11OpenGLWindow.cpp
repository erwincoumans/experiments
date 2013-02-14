#include "X11OpenGLWindow.h"
#include<stdio.h>
#include<stdlib.h>
#include<X11/X.h>
#include<X11/Xlib.h>
#include<GL/gl.h>
#include<GL/glx.h>
#include<GL/glu.h>


GLint                   att[] = { GLX_RGBA, GLX_DEPTH_SIZE, 24, GLX_DOUBLEBUFFER, None };


struct InternalData2
{
    Display*                m_dpy;
    Window                  m_root;
    XVisualInfo*            m_vi;
    Colormap                m_cmap;
    XSetWindowAttributes    m_swa;
    Window                  m_win;
    GLXContext              m_glc;
    XWindowAttributes       m_gwa;
    XEvent                  m_xev;
};


#if 0
int main2(int argc, char *argv[]) {


 glEnable(GL_DEPTH_TEST);

 while(1) {
 	XNextEvent(m_data->m_dpy, &m_data->m_xev);

        if(m_data->m_xev.type == Expose) {
        	XGetWindowAttributes(m_data->m_dpy, m_data->m_win, &m_data->m_gwa);
                glViewport(0, 0, m_data->m_gwa.width, m_data->m_gwa.height);
        	DrawAQuad();
                glXSwapBuffers(m_data->m_dpy, m_data->m_win);
        }

	else if(m_data->m_xev.type == KeyPress) {


 		exit(0);
        }
    } /* this closes while(1) { */
} /* this is the } which closes int main(int argc, char *argv[]) { */
#endif






























X11OpenGLWindow::X11OpenGLWindow()
:m_OpenGLInitialized(false)
{
    m_data = new InternalData2;
}

X11OpenGLWindow::~X11OpenGLWindow()
{
    if (m_OpenGLInitialized)
    {
        disableOpenGL();
    }

    delete m_data;
}



void X11OpenGLWindow::enableOpenGL()
{

    m_data->m_glc = glXCreateContext(m_data->m_dpy, m_data->m_vi, NULL, GL_TRUE);
    glXMakeCurrent(m_data->m_dpy, m_data->m_win, m_data->m_glc);

    const GLubyte* ven = glGetString(GL_VENDOR);
    printf("GL_VENDOR=%s\n", ven);
    const GLubyte* ren = glGetString(GL_RENDERER);
    printf("GL_RENDERER=%s\n",ren);
    const GLubyte* ver = glGetString(GL_VERSION);
    printf("GL_VERSION=%s\n", ver);
    const GLubyte* sl = glGetString(GL_SHADING_LANGUAGE_VERSION);
    printf("GL_SHADING_LANGUAGE_VERSION=%s\n", sl);
    const GLubyte* ext = glGetString(GL_EXTENSIONS);
    printf("GL_EXTENSIONS=%s\n", ext);
}

void X11OpenGLWindow::disableOpenGL()
{
    glXMakeCurrent(m_data->m_dpy, None, NULL);
 	glXDestroyContext(m_data->m_dpy, m_data->m_glc);
}


void    X11OpenGLWindow::createWindow(const btgWindowConstructionInfo& ci)
{
    m_data->m_dpy = XOpenDisplay(NULL);

    if(m_data->m_dpy == NULL) {
        printf("\n\tcannot connect to X server\n\n");
            exit(0);
     }

     m_data->m_root = DefaultRootWindow(m_data->m_dpy);

     m_data->m_vi = glXChooseVisual(m_data->m_dpy, 0, att);

     if(m_data->m_vi == NULL) {
        printf("\n\tno appropriate visual found\n\n");
            exit(0);
     }
     else {
        printf("\n\tvisual %p selected\n", (void *)m_data->m_vi->visualid); /* %p creates hexadecimal output like in glxinfo */
     }


     m_data->m_cmap = XCreateColormap(m_data->m_dpy, m_data->m_root, m_data->m_vi->visual, AllocNone);

     m_data->m_swa.colormap = m_data->m_cmap;
     m_data->m_swa.event_mask = ExposureMask | KeyPressMask;

     m_data->m_win = XCreateWindow(m_data->m_dpy, m_data->m_root, 0, 0, ci.m_width, ci.m_height, 0, m_data->m_vi->depth, InputOutput, m_data->m_vi->visual, CWColormap | CWEventMask, &m_data->m_swa);

     XMapWindow(m_data->m_dpy, m_data->m_win);
     XStoreName(m_data->m_dpy, m_data->m_win, "VERY SIMPLE APPLICATION");

    enableOpenGL();
}

void    X11OpenGLWindow::closeWindow()
{
    disableOpenGL();

    XDestroyWindow(m_data->m_dpy, m_data->m_win);
 	XCloseDisplay(m_data->m_dpy);
}


void    X11OpenGLWindow::startRendering()
{
//		pumpMessage();

    XGetWindowAttributes(m_data->m_dpy, m_data->m_win, &m_data->m_gwa);
    glViewport(0, 0, m_data->m_gwa.width, m_data->m_gwa.height);

	XNextEvent(m_data->m_dpy, &m_data->m_xev);

    if(m_data->m_xev.type == Expose) {
//        XGetWindowAttributes(m_data->m_dpy, m_data->m_win, &m_data->m_gwa);
//        glXSwapBuffers(m_data->m_dpy, m_data->m_win);
    }

	else if(m_data->m_xev.type == KeyPress)
	{
	    printf("key");
//        exit(0);
    }

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);	//clear buffers

    //glCullFace(GL_BACK);
    //glFrontFace(GL_CCW);
    glEnable(GL_DEPTH_TEST);
}

void    X11OpenGLWindow::renderAllObjects()
{

}

void    X11OpenGLWindow::endRendering()
{
    glXSwapBuffers(m_data->m_dpy, m_data->m_win);
}

void    X11OpenGLWindow::runMainLoop()
{

}

float   X11OpenGLWindow::getTimeInSeconds()
{
    return 0.f;
}

bool    X11OpenGLWindow::requestedExit() const
{
    return false;
}

void    X11OpenGLWindow::setRequestExit()
{

}


void X11OpenGLWindow::setMouseMoveCallback(btMouseMoveCallback   mouseCallback)
{

}

void X11OpenGLWindow::setMouseButtonCallback(btMouseButtonCallback       mouseCallback)
{

}

void X11OpenGLWindow::setResizeCallback(btResizeCallback resizeCallback)
{

}

void X11OpenGLWindow::setWheelCallback(btWheelCallback wheelCallback)
{

}

void X11OpenGLWindow::setKeyboardCallback( btKeyboardCallback    keyboardCallback)
{

}

void X11OpenGLWindow::setRenderCallback( btRenderCallback renderCallback)
{

}

void X11OpenGLWindow::setWindowTitle(const char* title)
{
    XStoreName(m_data->m_dpy, m_data->m_win, title);
}
