
#ifndef __OPENGL_INCLUDE_H
#define __OPENGL_INCLUDE_H

#include <GL/glew.h>

//think different
#if defined(__APPLE__) && !defined (VMDMESA)
#include <OpenGL/OpenGL.h>
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#else


#ifdef _WINDOWS
#include <windows.h>
#include <GL/gl.h>
#include <GL/glu.h>
#else
#include <GL/gl.h>

#endif //_WINDOWS
#endif //APPLE

#endif //__OPENGL_INCLUDE_H

