//
// Copyright (c) 2011 Andreas Krinke andreas.krinke@gmx.de
// Copyright (c) 2009 Mikko Mononen memon@inside.org
//
// This software is provided 'as-is', without any express or implied
// warranty.  In no event will the authors be held liable for any damages
// arising from the use of this software.
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
// 1. The origin of this software must not be misrepresented; you must not
//    claim that you wrote the original software. If you use this software
//    in a product, an acknowledgment in the product documentation would be
//    appreciated but is not required.
// 2. Altered source versions must be plainly marked as such, and must not be
//    misrepresented as being the original software.
// 3. This notice may not be removed or altered from any source distribution.
//

#define STB_TRUETYPE_IMPLEMENTATION
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <Windows.h>

#endif
#include "fontstash.h"

#ifdef __APPLE__
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define BORDER_X_LEFT 2
#define BORDER_X_RIGHT 2
#define BORDER_Y_TOP 2
#define BORDER_Y_BOTTOM 2
#define ADDITIONAL_HEIGHT 2

#define STB_TRUETYPE_IMPLEMENTATION
#define STBTT_malloc(x,u)    malloc(x)
#define STBTT_free(x,u)      free(x)
#include "stb_truetype.h"

#define HASH_LUT_SIZE 256
#define MAX_ROWS 128
#define VERT_COUNT (6*128)
#define INDEX_COUNT (VERT_COUNT*2)


#define TTFONT_FILE 1
#define TTFONT_MEM  2
#define BMFONT      3

static int idx = 1;

static unsigned int hashint(unsigned int a)
{
	a += ~(a<<15);
	a ^=  (a>>10);
	a +=  (a<<3);
	a ^=  (a>>6);
	a += ~(a<<11);
	a ^=  (a>>16);
	return a;
}


struct sth_quad
{
	float x0,y0,s0,t0;
	float x1,y1,s1,t1;
};

struct sth_row
{
	short x,y,h;
};

struct sth_glyph
{
	unsigned int codepoint;
	short size;
	struct sth_texture* texture;
	int x0_,y0,x1,y1;
	float xadv,xoff,yoff;
	int next;
};

struct sth_font
{
	int idx;
	int type;
	stbtt_fontinfo font;
	unsigned char* data;
	struct sth_glyph* glyphs;
	int lut[HASH_LUT_SIZE];
	int nglyphs;
	float ascender;
	float descender;
	float lineh;
	struct sth_font* next;
};

static unsigned int s_indexData[INDEX_COUNT];
GLuint s_indexArrayObject, s_indexBuffer;
GLuint s_vertexArrayObject,s_vertexBuffer;


struct sth_texture
{
	GLuint id;
    
	// TODO: replace rows with pointer
	struct sth_row rows[MAX_ROWS];
	int nrows;
	int nverts;
	
    Vertex newverts[VERT_COUNT];
  	struct sth_texture* next;
};

struct sth_stash
{
	int tw,th;
	float itw,ith;
	struct sth_texture* textures;
	struct sth_font* fonts;
	int drawing;
};



// Copyright (c) 2008-2009 Bjoern Hoehrmann <bjoern@hoehrmann.de>
// See http://bjoern.hoehrmann.de/utf-8/decoder/dfa/ for details.

#define UTF8_ACCEPT 0
#define UTF8_REJECT 1

static const unsigned char utf8d[] = {
	0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, // 00..1f
	0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, // 20..3f
	0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, // 40..5f
	0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, // 60..7f
	1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9, // 80..9f
	7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7, // a0..bf
	8,8,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2, // c0..df
	0xa,0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x4,0x3,0x3, // e0..ef
	0xb,0x6,0x6,0x6,0x5,0x8,0x8,0x8,0x8,0x8,0x8,0x8,0x8,0x8,0x8,0x8, // f0..ff
	0x0,0x1,0x2,0x3,0x5,0x8,0x7,0x1,0x1,0x1,0x4,0x6,0x1,0x1,0x1,0x1, // s0..s0
	1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,0,1,1,1,1,1,1, // s1..s2
	1,2,1,1,1,1,1,2,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1, // s3..s4
	1,2,1,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,3,1,3,1,1,1,1,1,1, // s5..s6
	1,3,1,1,1,1,1,3,1,3,1,1,1,1,1,1,1,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1, // s7..s8
};

static unsigned int decutf8(unsigned int* state, unsigned int* codep, unsigned int byte)
{
	unsigned int type = utf8d[byte];
	*codep = (*state != UTF8_ACCEPT) ?
		(byte & 0x3fu) | (*codep << 6) :
		(0xff >> type) & (byte);
	*state = utf8d[256 + *state*16 + type];
	return *state;
}



struct sth_stash* sth_create(int cachew, int cacheh)
{
	struct sth_stash* stash = NULL;
	struct sth_texture* texture = NULL;

	// Allocate memory for the font stash.
	stash = (struct sth_stash*)malloc(sizeof(struct sth_stash));
	if (stash == NULL)
    {
        assert(0);
        return NULL;
    }
	memset(stash,0,sizeof(struct sth_stash));

	// Allocate memory for the first texture
	texture = (struct sth_texture*)malloc(sizeof(struct sth_texture));
	if (texture == NULL)
    {
        assert(0);
        free(stash);
    }
    memset(texture,0,sizeof(struct sth_texture));

	// Create first texture for the cache.
	stash->tw = cachew;
	stash->th = cacheh;
	stash->itw = 1.0f/cachew;
	stash->ith = 1.0f/cacheh;
	stash->textures = texture;
	glGenTextures(1, &texture->id);
	if (!texture->id)
    {
        free(stash);
        free(texture);
        assert(0);
        return NULL;
    }
	glBindTexture(GL_TEXTURE_2D, texture->id);
	unsigned char* bmp = (unsigned char*)malloc(stash->tw*stash->th);
	memset(bmp,0,stash->tw*stash->th);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, stash->tw,stash->th, 0, GL_RED, GL_UNSIGNED_BYTE, bmp);
	free(bmp);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

    
    ////////////////////////////
    //create the other data
    {
        glGenVertexArrays(1, &s_vertexArrayObject);
        glBindVertexArray(s_vertexArrayObject);
        
        glGenBuffers(1, &s_vertexBuffer);
        glBindBuffer(GL_ARRAY_BUFFER, s_vertexBuffer);
        glBufferData(GL_ARRAY_BUFFER, VERT_COUNT * sizeof(Vertex), texture->newverts, GL_DYNAMIC_DRAW);
        GLuint err = glGetError();
        assert(err==GL_NO_ERROR);
        
        for (int i=0;i<INDEX_COUNT;i++)
        {
            s_indexData[i] = i;
        }
            
        glGenBuffers(1, &s_indexBuffer);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, s_indexBuffer);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,INDEX_COUNT*sizeof(int), s_indexData,GL_STATIC_DRAW);
        
        err = glGetError();
        assert(err==GL_NO_ERROR);
    }
    ////////////////////////////

    
    
    
    
    
    
	return stash;
	
}

int sth_add_font_from_memory(struct sth_stash* stash, unsigned char* buffer)
{
	int i, ascent, descent, fh, lineGap;
    GLint err;
	struct sth_font* fnt = NULL;

	fnt = (struct sth_font*)malloc(sizeof(struct sth_font));
	if (fnt == NULL) goto error;
	memset(fnt,0,sizeof(struct sth_font));

    err = glGetError();
    assert(err==GL_NO_ERROR);
    
	// Init hash lookup.
	for (i = 0; i < HASH_LUT_SIZE; ++i)
        fnt->lut[i] = -1;

	fnt->data = buffer;

	// Init stb_truetype
	if (!stbtt_InitFont(&fnt->font, fnt->data, 0))
        goto error;
	
    err = glGetError();
    assert(err==GL_NO_ERROR);
    
	// Store normalized line height. The real line height is got
	// by multiplying the lineh by font size.
	stbtt_GetFontVMetrics(&fnt->font, &ascent, &descent, &lineGap);
    err = glGetError();
    assert(err==GL_NO_ERROR);

	fh = ascent - descent;
	fnt->ascender = (float)ascent / (float)fh;
	fnt->descender = (float)descent / (float)fh;
	fnt->lineh = (float)(fh + lineGap) / (float)fh;

	fnt->idx = idx;
	fnt->type = TTFONT_MEM;
	fnt->next = stash->fonts;
	stash->fonts = fnt;
	
	return idx++;

error:
	if (fnt) {
		if (fnt->glyphs) free(fnt->glyphs);
		free(fnt);
	}
	return 0;
}

int sth_add_font(struct sth_stash* stash, const char* path)
{
	FILE* fp = 0;
	int datasize;
	unsigned char* data = NULL;
	int idx;
	
	// Read in the font data.
	fp = fopen(path, "rb");
	if (!fp) goto error;
	fseek(fp,0,SEEK_END);
	datasize = (int)ftell(fp);
	fseek(fp,0,SEEK_SET);
	data = (unsigned char*)malloc(datasize);
	if (data == NULL) goto error;
	fread(data, 1, datasize, fp);
	fclose(fp);
	fp = 0;
	
	idx = sth_add_font_from_memory(stash, data);
	// Modify type of the loaded font.
	if (idx)
		stash->fonts->type = TTFONT_FILE;
	else
		free(data);

	return idx;
	
error:
	if (data) free(data);
	if (fp) fclose(fp);
	return 0;
}

int sth_add_bitmap_font(struct sth_stash* stash, int ascent, int descent, int line_gap)
{
	int i, fh;
	struct sth_font* fnt = NULL;

	fnt = (struct sth_font*)malloc(sizeof(struct sth_font));
	if (fnt == NULL) goto error;
	memset(fnt,0,sizeof(struct sth_font));

	// Init hash lookup.
	for (i = 0; i < HASH_LUT_SIZE; ++i) fnt->lut[i] = -1;

	// Store normalized line height. The real line height is got
	// by multiplying the lineh by font size.
	fh = ascent - descent;
	fnt->ascender = (float)ascent / (float)fh;
	fnt->descender = (float)descent / (float)fh;
	fnt->lineh = (float)(fh + line_gap) / (float)fh;

	fnt->idx = idx;
	fnt->type = BMFONT;
	fnt->next = stash->fonts;
	stash->fonts = fnt;
	
	return idx++;

error:
	if (fnt) free(fnt);
	return 0;
}

void sth_add_glyph(struct sth_stash* stash,
                  int idx,
                  GLuint id,
                  const char* s,
                  short size, short base,
                  int x, int y, int w, int h,
                  float xoffset, float yoffset, float xadvance)
{
	struct sth_texture* texture = NULL;
	struct sth_font* fnt = NULL;
	struct sth_glyph* glyph = NULL;
	unsigned int codepoint;
	unsigned int state = 0;

	if (stash == NULL) return;
	texture = stash->textures;
	while (texture != NULL && texture->id != id) texture = texture->next;
	if (texture == NULL)
	{
		// Create new texture
		texture = (struct sth_texture*)malloc(sizeof(struct sth_texture));
		if (texture == NULL) return;
		memset(texture, 0, sizeof(struct sth_texture));
		texture->id = id;
		texture->next = stash->textures;
		stash->textures = texture;
	}

	fnt = stash->fonts;
	while (fnt != NULL && fnt->idx != idx) fnt = fnt->next;
	if (fnt == NULL) return;
	if (fnt->type != BMFONT) return;
	
	for (; *s; ++s)
	{
		if (!decutf8(&state, &codepoint, *(unsigned char*)s)) break;
	}
	if (state != UTF8_ACCEPT) return;

	// Alloc space for new glyph.
	fnt->nglyphs++;
	fnt->glyphs = (sth_glyph*)realloc(fnt->glyphs, fnt->nglyphs*sizeof(struct sth_glyph));
	if (!fnt->glyphs) return;

	// Init glyph.
	glyph = &fnt->glyphs[fnt->nglyphs-1];
	memset(glyph, 0, sizeof(struct sth_glyph));
	glyph->codepoint = codepoint;
	glyph->size = size;
	glyph->texture = texture;
	glyph->x0_ = x;
	glyph->y0 = y;
	glyph->x1 = glyph->x0_+w;
	glyph->y1 = glyph->y0+h;
	glyph->xoff = xoffset;
	glyph->yoff = yoffset - base;
	glyph->xadv = xadvance;
	
	// Find code point and size.
	h = hashint(codepoint) & (HASH_LUT_SIZE-1);
	// Insert char to hash lookup.
	glyph->next = fnt->lut[h];
	fnt->lut[h] = fnt->nglyphs-1;
}

static struct sth_glyph* get_glyph(struct sth_stash* stash, struct sth_font* fnt, unsigned int codepoint, short isize)
{
	int i,g,advance,lsb,x0,y0,x1,y1,gw,gh;
	float scale;
	struct sth_texture* texture = NULL;
	struct sth_glyph* glyph = NULL;
	unsigned char* bmp = NULL;
	unsigned int h;
	float size = isize/10.0f;
	int rh;
	struct sth_row* br = NULL;

	// Find code point and size.
	h = hashint(codepoint) & (HASH_LUT_SIZE-1);
	i = fnt->lut[h];
	while (i != -1)
	{
		if (fnt->glyphs[i].codepoint == codepoint && (fnt->type == BMFONT || fnt->glyphs[i].size == isize))
			return &fnt->glyphs[i];
		i = fnt->glyphs[i].next;
	}
	// Could not find glyph.
	
	// For bitmap fonts: ignore this glyph.
	if (fnt->type == BMFONT) return 0;
	
	// For truetype fonts: create this glyph.
	scale = stbtt_ScaleForPixelHeight(&fnt->font, size);
	g = stbtt_FindGlyphIndex(&fnt->font, codepoint);
	stbtt_GetGlyphHMetrics(&fnt->font, g, &advance, &lsb);
	stbtt_GetGlyphBitmapBox(&fnt->font, g, scale,scale, &x0,&y0,&x1,&y1);
	gw = x1-x0;
	gh = y1-y0;
	
    // Check if glyph is larger than maximum texture size
	if (gw >= stash->tw || gh >= stash->th)
		return 0;

	// Find texture and row where the glyph can be fit.
	br = NULL;
	rh = (gh+7) & ~7;
	texture = stash->textures;
	while(br == NULL)
	{
		for (i = 0; i < texture->nrows; ++i)
		{
			if (texture->rows[i].h >= rh && texture->rows[i].x+gw+1 <= stash->tw)
				br = &texture->rows[i];
		}
	
		// If no row is found, there are 3 possibilities:
		//   - add new row
		//   - try next texture
		//   - create new texture
		if (br == NULL)
		{
			short py = BORDER_Y_TOP;
			// Check that there is enough space.
			if (texture->nrows)
			{
				py = texture->rows[texture->nrows-1].y + texture->rows[texture->nrows-1].h+1;
				if (py+rh > stash->th)
				{
					if (texture->next != NULL)
					{
						texture = texture->next;
					}
					else
					{
						// Create new texture
						texture->next = (struct sth_texture*)malloc(sizeof(struct sth_texture));
						texture = texture->next;
						if (texture == NULL) goto error;
						memset(texture,0,sizeof(struct sth_texture));
						glGenTextures(1, &texture->id);
						if (!texture->id) goto error;
						glBindTexture(GL_TEXTURE_2D, texture->id);
						glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, stash->tw,stash->th, 0, GL_RED, GL_UNSIGNED_BYTE, 0);
						glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                        
					}
					continue;
				}
			}
			// Init and add row
			br = &texture->rows[texture->nrows];
			br->x = BORDER_X_LEFT;
			br->y = py+BORDER_Y_BOTTOM;
			br->h = rh+ADDITIONAL_HEIGHT;
			texture->nrows++;	
		}
	}
	
	// Alloc space for new glyph.
	fnt->nglyphs++;
	fnt->glyphs = (sth_glyph*)realloc(fnt->glyphs, fnt->nglyphs*sizeof(struct sth_glyph));
	if (!fnt->glyphs) return 0;

	// Init glyph.
	glyph = &fnt->glyphs[fnt->nglyphs-1];
	memset(glyph, 0, sizeof(struct sth_glyph));
	glyph->codepoint = codepoint;
	glyph->size = isize;
	glyph->texture = texture;
	glyph->x0_ = br->x;
	glyph->y0 = br->y;
	glyph->x1 = glyph->x0_+gw;
	glyph->y1 = glyph->y0+gh;
	glyph->xadv = scale * advance;
	glyph->xoff = (float)x0;
	glyph->yoff = (float)y0;
	glyph->next = 0;

	// Advance row location.
	br->x += gw+BORDER_X_RIGHT;

	// Insert char to hash lookup.
	glyph->next = fnt->lut[h];
	fnt->lut[h] = fnt->nglyphs-1;

	// Rasterize
	bmp = (unsigned char*)malloc(gw*gh);
	if (bmp)
	{
		memset(bmp,255,gw*gh);
		stbtt_MakeGlyphBitmap(&fnt->font, bmp, gw,gh,gw, scale,scale, g);
		// Update texture
		glBindTexture(GL_TEXTURE_2D, texture->id);
		glPixelStorei(GL_UNPACK_ALIGNMENT,1);
		glTexSubImage2D(GL_TEXTURE_2D, 0, glyph->x0_,glyph->y0, gw,gh, GL_RED,GL_UNSIGNED_BYTE,bmp);
		free(bmp);
	}
	
	return glyph;

error:
	if (texture)
		free(texture);
	return 0;
}

static int get_quad(struct sth_stash* stash, struct sth_font* fnt, struct sth_glyph* glyph, short isize, float* x, float* y, struct sth_quad* q)
{
	int rx,ry;
	float scale = 1.0f;

	if (fnt->type == BMFONT) scale = isize/(glyph->size*10.0f);

	rx = floorf(*x + scale * glyph->xoff);
	ry = floorf(*y - scale * glyph->yoff);
	
	q->x0 = rx;
	q->y0 = ry;
	q->x1 = rx + scale * (glyph->x1 - glyph->x0_);
	q->y1 = ry - scale * (glyph->y1 - glyph->y0);
	
	q->s0 = float(glyph->x0_) * stash->itw;
	q->t0 = float(glyph->y0) * stash->ith;
	q->s1 = float(glyph->x1) * stash->itw;
	q->t1 = float(glyph->y1) * stash->ith;
	
	*x += scale * glyph->xadv;
	
	return 1;
}

static Vertex* setv(Vertex* v, float x, float y, float s, float t, float width, float height)
{
	bool scale=true;
	if (scale)
	{
		v->position.p[0] = (x-width)/(width);
		v->position.p[1] = (y)/(height);
	} else
	{
		v->position.p[0] = (x-width/2)/(width/2);
		v->position.p[1] = (y-height/2)/(height/2);
	}
    v->position.p[2] = 0.f;
    v->position.p[3] = 1.f;
    
	v->uv.p[0] = s;
    v->uv.p[1] = t;

    v->colour.p[0] = 0.f;
    v->colour.p[1] = 0.f;
    v->colour.p[2] = 0.f;
    v->colour.p[3] = 1.f;
    
	return v+1;
}


extern   GLuint m_shaderProg;
extern    GLint m_positionUniform;
extern    GLint m_colourAttribute;
extern    GLint m_positionAttribute;
extern    GLint m_textureAttribute;
extern    GLuint m_vertexBuffer;
extern    GLuint m_vertexArrayObject;
extern    GLuint  m_indexBuffer;
extern    GLuint m_texturehandle;




void display2() {
    
    GLint err = glGetError();
    assert(err==GL_NO_ERROR);
   // glViewport(0,0,10,10);
    
	const float timeScale = 0.008f;
	
    glUseProgram(m_shaderProg);
    glBindBuffer(GL_ARRAY_BUFFER, s_vertexBuffer);
    glBindVertexArray(s_vertexArrayObject);
    
    err = glGetError();
    assert(err==GL_NO_ERROR);
    
    
    //   glBindTexture(GL_TEXTURE_2D,m_texturehandle);
    
    
    err = glGetError();
    assert(err==GL_NO_ERROR);
    
    vec2 p( 0.f,0.f);//?b?0.5f * sinf(timeValue), 0.5f * cosf(timeValue) );
    glUniform2fv(m_positionUniform, 1, (const GLfloat *)&p);
    
    err = glGetError();
    assert(err==GL_NO_ERROR);
	err = glGetError();
    assert(err==GL_NO_ERROR);
    
    glEnableVertexAttribArray(m_positionAttribute);
	err = glGetError();
    assert(err==GL_NO_ERROR);
    
    glEnableVertexAttribArray(m_colourAttribute);
	err = glGetError();
    assert(err==GL_NO_ERROR);
    
	glEnableVertexAttribArray(m_textureAttribute);
    
    glVertexAttribPointer(m_positionAttribute, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), (const GLvoid *)0);
    glVertexAttribPointer(m_colourAttribute  , 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), (const GLvoid *)sizeof(vec4));
    glVertexAttribPointer(m_textureAttribute , 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (const GLvoid *)(sizeof(vec4)+sizeof(vec4)));
	err = glGetError();
    assert(err==GL_NO_ERROR);
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_indexBuffer);
    
    //glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    int indexCount = 6;
    err = glGetError();
    assert(err==GL_NO_ERROR);
    
    glDrawElements(GL_TRIANGLES, indexCount, GL_UNSIGNED_INT, 0);
    err = glGetError();
    assert(err==GL_NO_ERROR);
    
    //	glutSwapBuffers();
}


static void flush_draw(struct sth_stash* stash)
{
	struct sth_texture* texture = stash->textures;
	while (texture)
	{
		if (texture->nverts > 0)
		{
	         //   display2();

				GLint err;
			    glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, texture->id);
            err = glGetError();
            assert(err==GL_NO_ERROR);
          // glBindBuffer(GL_ARRAY_BUFFER, s_vertexBuffer);
           // glBindVertexArray(s_vertexArrayObject);
            glBufferData(GL_ARRAY_BUFFER, texture->nverts * sizeof(Vertex), &texture->newverts[0].position.p[0], GL_DYNAMIC_DRAW);
            err = glGetError();
            assert(err==GL_NO_ERROR);
            
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, s_indexBuffer);
            
            //glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
            int indexCount = texture->nverts;
            err = glGetError();
            assert(err==GL_NO_ERROR);
            
            glDrawElements(GL_TRIANGLES, indexCount, GL_UNSIGNED_INT, 0);
            err = glGetError();
            assert(err==GL_NO_ERROR);
		
			texture->nverts = 0;
		}
		texture = texture->next;
	}
}


void sth_begin_draw(struct sth_stash* stash)
{
	if (stash == NULL) return;
	if (stash->drawing)
		flush_draw(stash);
	stash->drawing = 1;
}

void sth_end_draw(struct sth_stash* stash)
{
	if (stash == NULL) return;
	if (!stash->drawing) return;

/*
	// Debug dump.
	if (stash->nverts+6 < VERT_COUNT)
	{
		float x = 500, y = 100;
		float* v = &stash->verts[stash->nverts*4];

		v = setv(v, x, y, 0, 0);
		v = setv(v, x+stash->tw, y, 1, 0);
		v = setv(v, x+stash->tw, y+stash->th, 1, 1);

		v = setv(v, x, y, 0, 0);
		v = setv(v, x+stash->tw, y+stash->th, 1, 1);
		v = setv(v, x, y+stash->th, 0, 1);

		stash->nverts += 6;
	}
*/
	
	flush_draw(stash);
	stash->drawing = 0;
}

void sth_draw_texture(struct sth_stash* stash,
				   int idx, float size,
				   float x, float y,
				   int screenwidth, int screenheight,
				   const char* s, float* dx)
{
	int width = stash->tw;
	int height=stash->th;

	unsigned int codepoint;
	struct sth_glyph* glyph = NULL;
	struct sth_texture* texture = NULL;
	unsigned int state = 0;
	struct sth_quad q;
	short isize = (short)(size*10.0f);
	Vertex* v;
	struct sth_font* fnt = NULL;
	
	if (stash == NULL) return;

	if (!stash->textures) return;
	fnt = stash->fonts;
	while(fnt != NULL && fnt->idx != idx) fnt = fnt->next;
	if (fnt == NULL) return;
	if (fnt->type != BMFONT && !fnt->data) return;
	
	int once = true;
	for (; once; ++s)
	{
		once=false;
		if (decutf8(&state, &codepoint, *(unsigned char*)s)) 
			continue;
		glyph = get_glyph(stash, fnt, codepoint, isize);
		if (!glyph) 
			continue;
		texture = glyph->texture;
		if (texture->nverts+6 >= VERT_COUNT)
			flush_draw(stash);
		
		if (!get_quad(stash, fnt, glyph, isize, &x, &y, &q)) 
			continue;
		
		v = &texture->newverts[texture->nverts];
		q.x0 = 0;
		q.y0 = 0;
		q.x1 = q.x0+width;
		q.y1 = q.y0+height;

		v = setv(v, q.x0, q.y0, 0,0,screenwidth,screenheight);
		v = setv(v, q.x1, q.y0, 1,0,screenwidth,screenheight);
		v = setv(v, q.x1, q.y1, 1,1,screenwidth,screenheight);

		v = setv(v, q.x0, q.y0, 0,0,screenwidth,screenheight);
		v = setv(v, q.x1, q.y1, 1,1,screenwidth,screenheight);
		v = setv(v, q.x0, q.y1, 0,1,screenwidth,screenheight);
	


/*		v = setv(v, q.x0, q.y0, q.s0, q.t0);
		v = setv(v, q.x1, q.y0, q.s1, q.t0);
		v = setv(v, q.x1, q.y1, q.s1, q.t1);

		v = setv(v, q.x0, q.y0, q.s0, q.t0);
		v = setv(v, q.x1, q.y1, q.s1, q.t1);
		v = setv(v, q.x0, q.y1, q.s0, q.t1);
	*/	
		texture->nverts += 6;
	}
	
	flush_draw(stash);
#if 0
	glBindTexture(GL_TEXTURE_2D, texture->id);


	unsigned char* pixels = (unsigned char*)malloc(width*height);
	glReadPixels(0,0,width, height, GL_RED, GL_UNSIGNED_BYTE, pixels);
	//swap the pixels
	unsigned char* tmp = (unsigned char*)malloc(width);
	for (int j=0;j<height;j++)
	{
			pixels[j*width+j]=255;
	}
	if (0)
	{
		for (int j=0;j<height/2;j++)
		{
			for (int i=0;i<width;i++)
			{
				tmp[i] = pixels[j*width+i];
				pixels[j*width+i]=pixels[(height-j-1)*width+i];
				pixels[(height-j-1)*width+i] = tmp[i];
			}
			}
	}
	
	glBindTexture(GL_TEXTURE_2D, 0);
	GLint err = glGetError();
	if (err!=GL_NO_ERROR)
		exit(0);
	int comp=1;//1=Y
	stbi_write_png("fontmap.png", width,height, comp, pixels, width);
	
	free(pixels);
#endif

	if (dx) *dx = x;
}

void sth_flush_draw(struct sth_stash* stash)
{
	flush_draw(stash);
}
void sth_draw_text(struct sth_stash* stash,
				   int idx, float size,
				   float x, float y,
				   const char* s, float* dx, int screenwidth, int screenheight)
{
	unsigned int codepoint;
	struct sth_glyph* glyph = NULL;
	struct sth_texture* texture = NULL;
	unsigned int state = 0;
	struct sth_quad q;
	short isize = (short)(size*10.0f);
	Vertex* v;
	struct sth_font* fnt = NULL;
	
	if (stash == NULL) return;

	if (!stash->textures) return;
	fnt = stash->fonts;
	while(fnt != NULL && fnt->idx != idx) fnt = fnt->next;
	if (fnt == NULL) return;
	if (fnt->type != BMFONT && !fnt->data) return;
	
	for (; *s; ++s)
	{
		if (decutf8(&state, &codepoint, *(unsigned char*)s)) 
			continue;
		glyph = get_glyph(stash, fnt, codepoint, isize);
		if (!glyph) continue;
		texture = glyph->texture;
		if (texture->nverts+6 >= VERT_COUNT)
			flush_draw(stash);
		
		if (!get_quad(stash, fnt, glyph, isize, &x, &y, &q)) continue;
		
		v = &texture->newverts[texture->nverts];
		
		v = setv(v, q.x0, q.y0, q.s0, q.t0,screenwidth,screenheight);
		v = setv(v, q.x1, q.y0, q.s1, q.t0,screenwidth,screenheight);
		v = setv(v, q.x1, q.y1, q.s1, q.t1,screenwidth,screenheight);

		v = setv(v, q.x0, q.y0, q.s0, q.t0,screenwidth,screenheight);
		v = setv(v, q.x1, q.y1, q.s1, q.t1,screenwidth,screenheight);
		v = setv(v, q.x0, q.y1, q.s0, q.t1,screenwidth,screenheight);
		
		texture->nverts += 6;
	}
	
	if (dx) *dx = x;
}

void sth_dim_text(struct sth_stash* stash,
				  int idx, float size,
				  const char* s,
				  float* minx, float* miny, float* maxx, float* maxy)
{
	unsigned int codepoint;
	struct sth_glyph* glyph = NULL;
	unsigned int state = 0;
	struct sth_quad q;
	short isize = (short)(size*10.0f);
	struct sth_font* fnt = NULL;
	float x = 0, y = 0;
	
	if (stash == NULL) return;
	if (!stash->textures || !stash->textures->id) return;
	fnt = stash->fonts;
	while(fnt != NULL && fnt->idx != idx) fnt = fnt->next;
	if (fnt == NULL) return;
	if (fnt->type != BMFONT && !fnt->data) return;
	
	*minx = *maxx = x;
	*miny = *maxy = y;

	for (; *s; ++s)
	{
		if (decutf8(&state, &codepoint, *(unsigned char*)s)) continue;
		glyph = get_glyph(stash, fnt, codepoint, isize);
		if (!glyph) continue;
		if (!get_quad(stash, fnt, glyph, isize, &x, &y, &q)) continue;
		if (q.x0 < *minx) *minx = q.x0;
		if (q.x1 > *maxx) *maxx = q.x1;
		if (q.y1 < *miny) *miny = q.y1;
		if (q.y0 > *maxy) *maxy = q.y0;
	}
}

void sth_vmetrics(struct sth_stash* stash,
				  int idx, float size,
				  float* ascender, float* descender, float* lineh)
{
	struct sth_font* fnt = NULL;

	if (stash == NULL) return;
	if (!stash->textures || !stash->textures->id) return;
	fnt = stash->fonts;
	while(fnt != NULL && fnt->idx != idx) fnt = fnt->next;
	if (fnt == NULL) return;
	if (fnt->type != BMFONT && !fnt->data) return;
	if (ascender)
		*ascender = fnt->ascender*size;
	if (descender)
		*descender = fnt->descender*size;
	if (lineh)
		*lineh = fnt->lineh*size;
}

void sth_delete(struct sth_stash* stash)
{
	struct sth_texture* tex = NULL;
	struct sth_texture* curtex = NULL;
	struct sth_font* fnt = NULL;
	struct sth_font* curfnt = NULL;

	if (!stash) return;

	tex = stash->textures;
	while(tex != NULL) {
		curtex = tex;
		tex = tex->next;
		if (curtex->id)
			glDeleteTextures(1, &curtex->id);
		free(curtex);
	}

	fnt = stash->fonts;
	while(fnt != NULL) {
		curfnt = fnt;
		fnt = fnt->next;
		if (curfnt->glyphs)
			free(curfnt->glyphs);
		if (curfnt->type == TTFONT_FILE && curfnt->data)
			free(curfnt->data);
		free(curfnt);
	}
	free(stash);
}
