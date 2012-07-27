#ifndef MAC_OPENGL_WINDOW_H
#define MAC_OPENGL_WINDOW_H

class MacOpenGLWindow
{
    struct MacOpenGLWindowInternalData* m_internalData;
    float m_mouseX;
    float m_mouseY;
    
public:
    
    MacOpenGLWindow();
    virtual ~MacOpenGLWindow();
    
    void init(int width, int height);

    void exit();
    
    void startRendering();
    
    void endRendering();//swap buffers
    
    bool requestedExit();
    
    void getMouseCoordinates(int& x, int& y);
    
    void runMainLoop();

};


#endif

