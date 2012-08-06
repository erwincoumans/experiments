#ifndef MAC_OPENGL_WINDOW_H
#define MAC_OPENGL_WINDOW_H

typedef void (*btMouseCallback)(int button, int state, float x, float y);
typedef void (*btKeyboardCallback)(unsigned char key, int x, int y);


class MacOpenGLWindow
{
    struct MacOpenGLWindowInternalData* m_internalData;
    float m_mouseX;
    float m_mouseY;
    
    btMouseCallback m_mouseCallback;
    btKeyboardCallback m_keyboardCallback;
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
    
    void setMouseCallback(btMouseCallback	mouseCallback)
    {
        m_mouseCallback = mouseCallback;
    }
    
	void setKeyboardCallback( btKeyboardCallback	keyboardCallback)
    {
        m_keyboardCallback = keyboardCallback;
    }


};


#endif

