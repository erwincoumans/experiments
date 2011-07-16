#ifndef OPENTISSUE_UTILITY_GL_ON_SCREEN_DISPLAY_H
#define OPENTISSUE_UTILITY_GL_ON_SCREEN_DISPLAY_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

#define OSD_CUSTOM 0
#define OSD_TOP_LEFT 1
#define OSD_TOP_RIGHT 2
#define OSD_BOTTOM_LEFT 3
#define OSD_BOTTOM_RIGHT 4
#define OSD_CENTER 5

#include <vector>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <string>
#include <sstream>
#include <stdarg.h>

#ifdef DEFINE_GLUI
#include <GL/glui.h>
#else
#include <GL/glut.h>
#endif

namespace OpenTissue
{
  namespace gl
  {
    /**
    * On Screen Display Class.
    * This class provides an easy interface for writing textual 2D
    * output on the screen. Uses screen pixel units.
    *
    */
    template<typename math_types>
    class OnScreenDisplay 
    {
    public:

      typedef typename math_types::real_type       real_type;

    protected:

      int m_pos;
      int m_pos_x;
      int m_pos_y;
      
      int m_screen_width;
      int m_screen_height;
      
      int m_display_width;
      int m_display_height;
      
      unsigned int m_maxlen; ///< Max chars present in a single string (only for monospaced fonts)

      float m_alpha;
      float m_red;
      float m_green;
      float m_blue;
      bool  m_visible;
      bool  m_static;

      std::vector< std::string > m_string;

    public:

      /**
      *
      * @param x  Horizontal pixel position of the top left corner of the display
      * @param y  Vertical pixel position of the top left corner of the display
      */
      OnScreenDisplay(int x, int y)
        : m_pos(0)
        , m_pos_x(x)
        , m_pos_y(y)
        , m_alpha(0.5f)
        , m_visible(true)
        , m_static(false)
      {
        set_color(0.1f, 0.15f, 0.19f);
      }

      /**
      *
      * @param m_pos  Semantic description of the display position
      * OSD_TOP_LEFT, OSD_TOP_RIGHT....
      */
      OnScreenDisplay(int pos)
        : m_pos(pos)
        , m_pos_x(0)
        , m_pos_y(0)
        , m_alpha(0.5)
        , m_maxlen(0)
        , m_visible(true) 
        , m_static(false)
      {
        set_color(0.1f, 0.15f, 0.19f); 
      }

      OnScreenDisplay()
        : m_pos(OSD_TOP_LEFT)
        , m_pos_x(0)
        , m_pos_y(0)
        , m_alpha(0.5)
        , m_maxlen(0)
        ,m_visible(true)
        ,m_static(false)
      {
        set_color(0.1f, 0.15f, 0.19f); 
      }
      
      /**
      * Sets the display position using semantic description predefines
      * of the display position i.e. OSD_TOP_LEFT, OSD_TOP_RIGHT ...
      *
      * @param m_pos  Semantic description predefine
      */
      void set_pos(int pos)
      {
        m_pos = pos;
        m_pos_x = m_pos_y = 0;
      }

      /**
      * Sets custom screen position of the OSD
      *
      * @param x  Horizontal pixel position of the top left corner of the display
      * @param y  Vertical pixel position of the top left corner of the display
      */
      void set_pos(int x, int y)
      {
        m_pos = 0;
        m_pos_x = x;
        m_pos_y = y;
      }

      void set_visibility(bool visible)
      {
        m_visible = visible;
      }

      bool get_visibility()
      {
        return m_visible;
      }

      void set_static(bool isstatic)
      {
        m_static = isstatic;
      }

      bool get_static()
      {
        return m_static;
      }

      void set_color(float r, float g, float b, float alpha=0.5f)
      {
        m_red=r; m_green=g; m_blue=b; m_alpha=alpha;
      }

      /**
      * Prints a formatted string to the OSD in standard printf format(see C++ docs)
      *
      * @param format The format string
      * @param ...    Variable length argument list for formatting
      */
      void printf(char * format, ...)
      {
        char buffer[256];
        va_list args;
        va_start (args, format);
        vsprintf (buffer,format, args);
        //perror (buffer);
        va_end (args);
        this->print(std::string(buffer));
      }

      void print(std::string const & str)
      {
        m_string.push_back(str);
        // Set the display width to accommodate the longest string
        if(str.length() > m_maxlen)
          m_maxlen = str.length();
      }

      void clear()
      {
        m_string.clear();
        m_maxlen = 0u;
      }

      void display()
      {
        if(!m_visible)
          return;

        int x_offset=0;
        int y_offset=0;

#ifdef DEFINE_GLUI
        GLUI_Master.get_viewport_area( &x_offset, &y_offset, &m_screen_width, &m_screen_height );
#elif DEFINE_GLUI_DISABLE
        m_screen_width = glutGet(GLUT_WINDOW_WIDTH);
        m_screen_height = glutGet(GLUT_WINDOW_HEIGHT);
#else
        m_screen_width = glutGet(GLUT_WINDOW_WIDTH);
        m_screen_height = glutGet(GLUT_WINDOW_HEIGHT);
#endif
        // Font width * maxlen + 2 frame pixels
        m_display_width =  2 + 8 * m_maxlen + 2;
        // 6 grace pixels str.size * (font height + line spacing)
        m_display_height = 6 + m_string.size()*(13 + 2); 

        switch(m_pos)
        {
        case  OSD_CUSTOM:
          m_pos_x += x_offset;
          m_pos_y += y_offset;
          if(m_pos_x > m_screen_width-m_display_width)
            m_pos_x = m_screen_width-m_display_width;
          else if(m_pos_x < x_offset)
            m_pos_x = x_offset;
          if(m_pos_y > m_screen_height-m_display_height)
            m_pos_y = m_screen_height-m_display_height;
          else if(m_pos_y < y_offset)
            m_pos_y = y_offset;
          break;
        case  OSD_TOP_LEFT: break;
        case  OSD_TOP_RIGHT:
          m_pos_x = m_screen_width-m_display_width;
          break;
        case  OSD_BOTTOM_LEFT:
          m_pos_y = y_offset + m_screen_height-m_display_height;
          break;
        case  OSD_BOTTOM_RIGHT:
          m_pos_x = m_screen_width-m_display_width;
          m_pos_y = m_screen_height-m_display_height;
          break;
        case  OSD_CENTER:
          m_pos_x = x_offset + ((m_screen_width/2)-(m_display_width/2));
          y_offset = 0;
          m_pos_y = y_offset + ((m_screen_height/2)-(m_display_height/2));
          break;
        default:
          break;
        }
        display_text();
        if(!m_static)
          clear();
      }//display

    protected:

      void renderBitmapString(float x, float y, float z, void *font, const char *string)
      {  
        const char *c;
        glRasterPos3f(m_pos_x + x, m_pos_y + y, z);
        for (c=string; *c != '\0'; c++) {
          glutBitmapCharacter(font, *c);
        }
      }

      void display_text()
      {
        if(m_alpha > 0.0 && m_alpha < 1.0)
        {
	        // Do 2D rendering
	        glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA); //_MINUS_SRC_ALPHA);
	        glEnable(GL_BLEND);
	        glDisable(GL_TEXTURE_2D);
        }
        glDisable(GL_DEPTH_TEST);
        //glDepthMask(1);
	      glDisable(GL_LIGHTING);
	      glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );
	      // Save GL state
	      glMatrixMode(GL_PROJECTION);	// Select The Projection Matrix
	      glPushMatrix();					// Store The Projection Matrix
	      glLoadIdentity();				// Reset The Projection Matrix
	      glOrtho(0,m_screen_width,m_screen_height,0,-10,10);	// Set Up An Ortho Screen
	      glMatrixMode(GL_MODELVIEW);		// Select The Modelview Matrix
	      glPushMatrix();					// Store The Modelview Matrix
	      glLoadIdentity();				// Reset The Modelview Matrix

        if(m_alpha > 0.0){
  	      gl::ColorPicker(m_red,m_green,m_blue,m_alpha);
          glBegin(GL_QUADS);						// Draw A Quad
	        glVertex3f( m_pos_x+0.0f, m_pos_y+0.0f, -2.1f);				// Top Left
	        glVertex3f( m_pos_x+m_display_width, m_pos_y+0.0f, -2.1f);				// Top Right
	        glVertex3f( m_pos_x+m_display_width, m_pos_y+m_display_height, -2.1f);				// Bottom Right
	        glVertex3f( m_pos_x+0.0f,   m_pos_y+m_display_height, -2.1f);				// Bottom Left
	        glEnd();							// Done Drawing The Quad
	        glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
	        gl::ColorPicker(0.9,0.9,0.9,1.0);
	        glBegin(GL_QUADS);						// Draw A Quad
	        glVertex3f( m_pos_x+0.0f, m_pos_y+0.0f, -2.0f);				// Top Left
	        glVertex3f( m_pos_x+m_display_width, m_pos_y+0.0f, -2.0f);				// Top Right
	        glVertex3f( m_pos_x+m_display_width, m_pos_y+m_display_height, -2.0f);				// Bottom Right
	        glVertex3f( m_pos_x+0.0f,   m_pos_y+m_display_height, -2.0f);				// Bottom Left
	        glEnd();							// Done Drawing The Quad
        }
	      gl::ColorPicker(0.9,0.9,0.9,1.0);
	      int count = 0;
        for (std::vector<std::string>::iterator it = m_string.begin(); it!=m_string.end(); ++it) 
        {
		      count++;
		      renderBitmapString(3,15*count,1,GLUT_BITMAP_8_BY_13,(*it).c_str());
	      }
	      glEnable(GL_LIGHTING);
	      glEnable(GL_DEPTH_TEST);
	      glDisable(GL_BLEND);
	      glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );

	      // Restore state
	      glMatrixMode(GL_PROJECTION);						// Select The Projection Matrix
	      glPopMatrix();								// Restore The Old Projection Matrix
	      glMatrixMode(GL_MODELVIEW);						// Select The Modelview Matrix
	      glPopMatrix();								// Restore The Old Projection Matrix
      }

    };

  } // namespace gl

} // namespace OpenTissue

// OPENTISSUE_UTILITY_GL_ON_SCREEN_DISPLAY_H

#endif
