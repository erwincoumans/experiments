#ifndef MAC_OPENGL_WINDOW_H
#define MAC_OPENGL_WINDOW_H

typedef void (*btMouseButtonCallback)(int button, int state, float x, float y);
typedef void (*btMouseMoveCallback)(float x, float y);
typedef void (*btResizeCallback)(float x, float y);
typedef void (*btKeyboardCallback)(unsigned char key, int x, int y);
typedef void (*btWheelCallback)(float deltax, float deltay);

class MacOpenGLWindow
{
    struct MacOpenGLWindowInternalData* m_internalData;
    float m_mouseX;
    float m_mouseY;

   
    btMouseButtonCallback m_mouseButtonCallback;
    btMouseMoveCallback m_mouseMoveCallback;
    btWheelCallback m_wheelCallback;
    btKeyboardCallback m_keyboardCallback;
    float m_retinaScaleFactor;
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
    
    void setMouseButtonCallback(btMouseButtonCallback	mouseCallback)
    {
        m_mouseButtonCallback = mouseCallback;
    }

    void setMouseMoveCallback(btMouseMoveCallback	mouseCallback)
    {
        m_mouseMoveCallback = mouseCallback;
    }
    
    void setResizeCallback(btResizeCallback resizeCallback);
    
	void setKeyboardCallback( btKeyboardCallback	keyboardCallback)
    {
        m_keyboardCallback = keyboardCallback;
    }
    
    void setWheelCallback (btWheelCallback wheelCallback)
    {
        m_wheelCallback = wheelCallback;
    }

    float getRetinaScale() const
    {
        return m_retinaScaleFactor;
    }

};


#endif

