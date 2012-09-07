#ifndef BTG_WINDOW_INTERFACE_H
#define BTG_WINDOW_INTERFACE_H


typedef void (*btWheelCallback)(float deltax, float deltay);
typedef void (*btResizeCallback)( float width, float height);
typedef void (*btMouseMoveCallback)( float x, float y);
typedef void (*btMouseButtonCallback)(int button, int state, float x, float y);
typedef void (*btKeyboardCallback)(unsigned char key, int x, int y);
typedef void (*btRenderCallback) ();

struct btgWindowConstructionInfo
{
		int m_width;
		int m_height;
		bool m_fullscreen;
		int m_colorBitsPerPixel;
		void* m_windowHandle;
		
		btgWindowConstructionInfo(int width=1024, int height=768)
		:m_width(width),
			m_height(height),
			m_fullscreen(false),
			m_colorBitsPerPixel(32),
			m_windowHandle(0)
			{
			}
};


class btgWindowInterface
{
	public:
		
		virtual ~btgWindowInterface()
		{
		}

		virtual	void	createWindow(const btgWindowConstructionInfo& ci)=0;
		
		virtual void	closeWindow()=0;

		virtual void	runMainLoop()=0;
		virtual	float	getTimeInSeconds()=0;
		
		virtual void setMouseMoveCallback(btMouseMoveCallback	mouseCallback)=0;
		virtual void setMouseButtonCallback(btMouseButtonCallback	mouseCallback)=0;
		virtual void setResizeCallback(btResizeCallback	resizeCallback)=0;
		virtual void setWheelCallback(btWheelCallback wheelCallback)=0;
		virtual void setKeyboardCallback( btKeyboardCallback	keyboardCallback)=0;

		virtual void setRenderCallback( btRenderCallback renderCallback) = 0;
	
		virtual void setWindowTitle(const char* title)=0;


};

#endif //BTG_WINDOW_INTERFACE_H
