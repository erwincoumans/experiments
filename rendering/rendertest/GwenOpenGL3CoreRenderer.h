
#ifndef __GWEN_OPENGL3_CORE_RENDERER_H
#define __GWEN_OPENGL3_CORE_RENDERER_H

#include "Gwen/Gwen.h"
#include "Gwen/BaseRender.h"
#include "GLPrimitiveRenderer.h"
struct sth_stash;
#include "../OpenGLTrueTypeFont/fontstash.h"


class GwenOpenGL3CoreRenderer : public Gwen::Renderer::Base
{
	GLPrimitiveRenderer* m_primitiveRenderer;
	float m_currentColor[4];
	float m_yOffset;
    sth_stash* m_font;
    float m_screenWidth;
    float m_screenHeight;
    float m_fontScaling;
    float m_retinaScale;
public:
	GwenOpenGL3CoreRenderer (GLPrimitiveRenderer* primRender, sth_stash* font,float screenWidth, float screenHeight, float retinaScale)
		:m_primitiveRenderer(primRender),
    m_font(font),
    m_screenWidth(screenWidth),
    m_screenHeight(screenHeight),
    m_retinaScale(retinaScale)
	{
		m_currentColor[0] = 1;
		m_currentColor[1] = 1;
		m_currentColor[2] = 1;
		m_currentColor[3] = 1;
        
        m_fontScaling = 15.f*m_retinaScale;
	}

	void resize(int width, int height)
	{
		m_screenWidth = width;
		m_screenHeight = height;
	}
	
	virtual void Begin()
	{
		m_yOffset=0;
	}
	virtual void End()
	{
	}

	virtual void StartClip()
	{

		sth_flush_draw(m_font);
		Gwen::Rect rect = ClipRegion();

		// OpenGL's coords are from the bottom left
		// so we need to translate them here.
		{
			GLint view[4];
			glGetIntegerv( GL_VIEWPORT, &view[0] );
			rect.y = view[3]/m_retinaScale - (rect.y + rect.h);
		}

		glScissor( m_retinaScale * rect.x * Scale(), m_retinaScale * rect.y * Scale(), m_retinaScale * rect.w * Scale(), m_retinaScale * rect.h * Scale() );
		glEnable( GL_SCISSOR_TEST );
	};

	virtual void EndClip()
	{
		sth_flush_draw(m_font);
		glDisable( GL_SCISSOR_TEST );
	};

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
    
    void RenderText( Gwen::Font* pFont, Gwen::Point pos, const Gwen::UnicodeString& text )
    {
        Gwen::String str = Gwen::Utility::UnicodeToString(text);
        const char* unicodeText = (const char*)str.c_str();
        
        Gwen::Rect r;
        r.x = pos.x;
        r.y = pos.y;
        r.w = 0;
        r.h = 0;
    
        Translate(r);
        
      //
        //printf("str = %s\n",unicodeText);
        int xpos=0;
        int ypos=0;
        float dx;
        
        int measureOnly=0;
        
        sth_draw_text(m_font,
                      1,m_fontScaling,
                      r.x,r.y,
                      unicodeText,&dx, m_screenWidth,m_screenHeight,measureOnly,m_retinaScale);
        
        
       // Gwen::Renderer::Base::RenderText(pFont,pos,text);

    }
    Gwen::Point MeasureText( Gwen::Font* pFont, const Gwen::UnicodeString& text )
    {
        Gwen::String str = Gwen::Utility::UnicodeToString(text);
        const char* unicodeText = (const char*)str.c_str();
        
       // printf("str = %s\n",unicodeText);
        int xpos=0;
        int ypos=0;
        float dx;
        
        int measureOnly=1;
        
        sth_draw_text(m_font,
                      1,m_fontScaling,
                      xpos,ypos,
                      unicodeText,&dx, m_screenWidth,m_screenHeight,measureOnly);
        Gwen::Point pt;
        pt.x = dx*Scale();
        pt.y = m_fontScaling*Scale();//*0.8f;
        return pt;
//        return Gwen::Renderer::Base::MeasureText(pFont,text);
    }


			
};
#endif //__GWEN_OPENGL3_CORE_RENDERER_H