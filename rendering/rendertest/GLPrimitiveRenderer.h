#ifndef _GL_PRIMITIVE_RENDERER_H
#define _GL_PRIMITIVE_RENDERER_H

#include "OpenGLInclude.h"

class GLPrimitiveRenderer
{
    int m_screenWidth;
    int m_screenHeight;

    GLuint m_shaderProg;
    GLint m_positionUniform;
    GLint m_colourAttribute;
    GLint m_positionAttribute;
    GLint m_textureAttribute;
    GLuint m_vertexBuffer;
    GLuint m_vertexArrayObject;
    GLuint  m_indexBuffer;
    GLuint m_texturehandle;
    
    void loadBufferData();
    
public:
    
    GLPrimitiveRenderer(int screenWidth, int screenHeight);
    virtual ~GLPrimitiveRenderer();
    
    void drawLine();//float from[4], float to[4], float color[4]);
    void setScreenSize(int width, int height);
    
};

#endif//_GL_PRIMITIVE_RENDERER_H

