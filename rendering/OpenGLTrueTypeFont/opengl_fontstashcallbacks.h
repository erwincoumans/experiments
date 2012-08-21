#ifndef _OPENGL_FONTSTASH_CALLBACKS_H
#define _OPENGL_FONTSTASH_CALLBACKS_H


void OpenGL2UpdateTextureCallback(sth_texture* texture, sth_glyph* glyph, int textureWidth, int textureHeight);

void OpenGL2RenderCallback(sth_texture* texture);

//temporarily display2 method to hookup shader program etc
void display2();

void dumpTextureToPng( int screenWidth, int screenHeight, const char* fileName);


#endif//_OPENGL_FONTSTASH_CALLBACKS_H

