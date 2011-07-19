
#include "Gwen/Renderers/OpenGL_DebugFont.h"
#include "Gwen/Utility.h"
#include "Gwen/Font.h"
#include "Gwen/Texture.h"

#include <math.h>
#include "GL/glew.h"

#include "FontData.h"


namespace Gwen
{
	namespace Renderer
	{
		OpenGL_DebugFont::OpenGL_DebugFont()
		{
			m_iVertNum = 0;

			for ( int i=0; i<MaxVerts; i++ )
			{
				m_Vertices[ i ].z = 0.5f;
			}

			m_fLetterSpacing = 1.0f/16.0f;
			m_fFontScale[0] = 1.5f;
			m_fFontScale[1] = 1.5f;

			m_pFontTexture = new Gwen::Texture();

			// Create a little texture pointer..
			GLuint* pglTexture = new GLuint;

			// Sort out our GWEN texture
			m_pFontTexture->data = pglTexture;
			m_pFontTexture->width = 256;
			m_pFontTexture->height = 256;


			// Create the opengl texture
			glGenTextures( 1, pglTexture );
			glBindTexture( GL_TEXTURE_2D, *pglTexture );
			//glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
			//glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
			glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
			glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

			GLenum format = GL_RGB;
			unsigned char* texdata = new unsigned char[256*256*4];
			for (int i=0;i<256*256;i++)
			{
				texdata[i*4] = sGwenFontData[i];
				texdata[i*4+1] = sGwenFontData[i];
				texdata[i*4+2] = sGwenFontData[i];
				texdata[i*4+3] = sGwenFontData[i];
			}
			glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, m_pFontTexture->width, m_pFontTexture->height, 0, GL_RGBA, GL_UNSIGNED_BYTE, (const GLvoid*)texdata );
			delete[]texdata;
		}

		OpenGL_DebugFont::~OpenGL_DebugFont()
		{
			FreeTexture( m_pFontTexture );
			delete m_pFontTexture;
		}

		void OpenGL_DebugFont::Begin()
		{
			glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
			glAlphaFunc( GL_GREATER, 1.0f );	
			glEnable ( GL_BLEND );
		}

		void OpenGL_DebugFont::End()
		{
			if ( m_iVertNum == 0 ) return;

			glVertexPointer( 3, GL_FLOAT,  sizeof(Vertex), (void*) &m_Vertices[0].x );
			glEnableClientState( GL_VERTEX_ARRAY );

			glColorPointer( 4, GL_UNSIGNED_BYTE, sizeof(Vertex), (void*)&m_Vertices[0].r );
			glEnableClientState( GL_COLOR_ARRAY );

			glTexCoordPointer( 2, GL_FLOAT, sizeof(Vertex), (void*) &m_Vertices[0].u );
			glEnableClientState( GL_TEXTURE_COORD_ARRAY );

			glDrawArrays( GL_TRIANGLES, 0, (GLsizei) m_iVertNum );

			m_iVertNum = 0;
			glFlush();
		}

		void OpenGL_DebugFont::Flush()
		{
			if ( m_iVertNum == 0 ) return;

			glVertexPointer( 3, GL_FLOAT,  sizeof(Vertex), (void*) &m_Vertices[0].x );
			glEnableClientState( GL_VERTEX_ARRAY );

			glColorPointer( 4, GL_UNSIGNED_BYTE, sizeof(Vertex), (void*)&m_Vertices[0].r );
			glEnableClientState( GL_COLOR_ARRAY );

			glTexCoordPointer( 2, GL_FLOAT, sizeof(Vertex), (void*) &m_Vertices[0].u );
			glEnableClientState( GL_TEXTURE_COORD_ARRAY );

			glDrawArrays( GL_TRIANGLES, 0, (GLsizei) m_iVertNum );

			m_iVertNum = 0;
			glFlush();
		}

		void OpenGL_DebugFont::AddVert( int x, int y, float u, float v )
		{
			if ( m_iVertNum >= MaxVerts-1 )
			{
				Flush();
			}

			m_Vertices[ m_iVertNum ].x = (float)x;
			m_Vertices[ m_iVertNum ].y = (float)y;
			m_Vertices[ m_iVertNum ].u = u;
			m_Vertices[ m_iVertNum ].v = v;

			m_Vertices[ m_iVertNum ].r = m_Color.r;
			m_Vertices[ m_iVertNum ].g = m_Color.g;
			m_Vertices[ m_iVertNum ].b = m_Color.b;
			m_Vertices[ m_iVertNum ].a = m_Color.a;

			m_iVertNum++;
		}

		void OpenGL_DebugFont::DrawFilledRect( Gwen::Rect rect )
		{
			GLboolean texturesOn;

			glGetBooleanv(GL_TEXTURE_2D, &texturesOn);
			if ( texturesOn )
			{
				Flush();
				glDisable(GL_TEXTURE_2D);
			}	

			Translate( rect );

			AddVert( rect.x, rect.y );
			AddVert( rect.x+rect.w, rect.y );
			AddVert( rect.x, rect.y + rect.h );

			AddVert( rect.x+rect.w, rect.y );
			AddVert( rect.x+rect.w, rect.y+rect.h );
			AddVert( rect.x, rect.y + rect.h );
		}

		void OpenGL_DebugFont::SetDrawColor(Gwen::Color color)
		{
			glColor4ubv( (GLubyte*)&color );
			m_Color = color;
		}

		void OpenGL_DebugFont::StartClip()
		{
			Flush();
			Gwen::Rect rect = ClipRegion();

			// OpenGL's coords are from the bottom left
			// so we need to translate them here.
			{
				GLint view[4];
				glGetIntegerv( GL_VIEWPORT, &view[0] );
				rect.y = view[3] - (rect.y + rect.h);
			}

			glScissor( rect.x * Scale(), rect.y * Scale(), rect.w * Scale(), rect.h * Scale() );
			glEnable( GL_SCISSOR_TEST );
		};

		void OpenGL_DebugFont::EndClip()
		{
			Flush();
			glDisable( GL_SCISSOR_TEST );
			
		};

		void OpenGL_DebugFont::RenderText( Gwen::Font* pFont, Gwen::Point pos, const Gwen::UnicodeString& text )
		{
			float fSize = pFont->size * Scale();

			if ( !text.length() )
				return;

			Gwen::String converted_string = Gwen::Utility::UnicodeToString( text );

			float yOffset=0.0f;
			for ( int i=0; i<text.length(); i++ )
			{
				wchar_t chr = text[i];
				char ch = converted_string[i];
				float curSpacing = sGwenDebugFontSpacing[ch] * m_fLetterSpacing * fSize * m_fFontScale[0];
				Gwen::Rect r( pos.x + yOffset, pos.y-fSize*0.2f, (fSize * m_fFontScale[0]), fSize * m_fFontScale[1] );

				if ( m_pFontTexture )
				{
					float uv_texcoords[8]={0.,0.,1.,1.};

					if ( ch >= 0 )
					{
						float cx= (ch%16)/16.0;
						float cy= (ch/16)/16.0;
						uv_texcoords[0] = cx;			
						uv_texcoords[1] = cy;
						uv_texcoords[4] = float(cx+1.0f/16.0f);	
						uv_texcoords[5] = float(cy+1.0f/16.0f);
					}

					DrawTexturedRect( m_pFontTexture, r, uv_texcoords[0], uv_texcoords[5], uv_texcoords[4], uv_texcoords[1] );
					yOffset+=curSpacing;
				} 
				else
				{
					DrawFilledRect( r );
					yOffset+=curSpacing;

				}
			}

		}

		void OpenGL_DebugFont::DrawTexturedRect( Gwen::Texture* pTexture, Gwen::Rect rect, float u1, float v1, float u2, float v2 )
		{
			GLuint* tex = (GLuint*)pTexture->data;

			// Missing image, not loaded properly?
			if ( !tex )
			{
				return DrawMissingImage( rect );
			}

			Translate( rect );
			GLuint boundtex;

			GLboolean texturesOn;
			glGetBooleanv(GL_TEXTURE_2D, &texturesOn);
			glGetIntegerv(GL_TEXTURE_BINDING_2D, (GLint *)&boundtex);
			if ( !texturesOn || *tex != boundtex )
			{
				Flush();
				glBindTexture( GL_TEXTURE_2D, *tex );
				glEnable(GL_TEXTURE_2D);
			}		

			AddVert( rect.x, rect.y,			u1, v1 );
			AddVert( rect.x+rect.w, rect.y,		u2, v1 );
			AddVert( rect.x, rect.y + rect.h,	u1, v2 );

			AddVert( rect.x+rect.w, rect.y,		u2, v1 );
			AddVert( rect.x+rect.w, rect.y+rect.h, u2, v2 );
			AddVert( rect.x, rect.y + rect.h, u1, v2 );			
		}

		Gwen::Point OpenGL_DebugFont::MeasureText( Gwen::Font* pFont, const Gwen::UnicodeString& text )
		{
			Gwen::Point p;
			float fSize = pFont->size * Scale();

			Gwen::String converted_string = Gwen::Utility::UnicodeToString( text );
			float spacing = 0.0f;

			for ( int i=0; i<text.length(); i++ )
			{
				char ch = converted_string[i];
				spacing += sGwenDebugFontSpacing[ch];
			}

			p.x = spacing*m_fLetterSpacing*fSize * m_fFontScale[0];
			p.y = pFont->size * Scale() * m_fFontScale[1];
			return p;
		}

	}
}