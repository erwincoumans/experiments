
#ifndef __GWEN_OPENGL3_CORE_RENDERER_H
#define __GWEN_OPENGL3_CORE_RENDERER_H

#include "Gwen/Gwen.h"
#include "Gwen/BaseRender.h"
#include "GLPrimitiveRenderer.h"


class GwenOpenGL3CoreRenderer : public Gwen::Renderer::Base
{
	GLPrimitiveRenderer* m_primitiveRenderer;
	float m_currentColor[4];
	float m_yOffset;
public:
	GwenOpenGL3CoreRenderer (GLPrimitiveRenderer* primRender)
		:m_primitiveRenderer(primRender)
	{
		m_currentColor[0] = 1;
		m_currentColor[1] = 1;
		m_currentColor[2] = 1;
		m_currentColor[3] = 1;
	}
	
	virtual void Begin()
	{
		m_yOffset=0;
	}
	virtual void End()
	{
	}

	virtual void SetDrawColor( Gwen::Color color )
	{
		m_currentColor[0] = color.r/256.f;
		m_currentColor[1] = color.g/256.f;
		m_currentColor[2] = color.b/256.f;
		m_currentColor[3] = color.a/256.f;

	}
	virtual void DrawFilledRect( Gwen::Rect rect )
	{
		Translate( rect );

		m_primitiveRenderer->drawRect(rect.x, rect.y+m_yOffset, rect.x+rect.w, rect.y+rect.h+m_yOffset, m_currentColor);
//		m_yOffset+=rect.h+10;

	}
			
};
#endif //__GWEN_OPENGL3_CORE_RENDERER_H