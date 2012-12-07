#ifndef _GWEN_USER_INTERFACE_H
#define _GWEN_USER_INTERFACE_H

struct GwenInternalData;

typedef void (*btToggleButtonCallback)(int button, int state);

class GwenUserInterface
{
	GwenInternalData*	m_data;

	public:
		
		GwenUserInterface();
		
		virtual ~GwenUserInterface();
		
		void	init(int width, int height,struct sth_stash* stash,float retinaScale);
		
		void	draw(int width, int height);

		void	resize(int width, int height);
				
		bool	mouseMoveCallback( float x, float y);
		bool	mouseButtonCallback(int button, int state, float x, float y);

		void	setToggleButtonCallback(btToggleButtonCallback callback);
		void	registerToggleButton(int buttonId, const char* name);
};

#endif //_GWEN_USER_INTERFACE_H

