/*
Oolong Engine for the iPhone / iPod touch
Copyright (c) 2007-2008 Wolfgang Engel  http://code.google.com/p/oolongengine/

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/
#ifndef MACROS_H_
#define MACROS_H_

//#define TOOLSDEBUGOUT(A) printf(A)
//#define FREE(X)		{ if(X) { free(X); (X) = 0; } }


#define PT_INDEX (2)	/*The Punch-through index*/

#define BLK_Y_SIZE 	(4) /*always 4 for all 2D block types*/

#define BLK_X_MAX	(8)	/*Max X dimension for blocks*/

#define BLK_X_2BPP	(8) /*dimensions for the two formats*/
#define BLK_X_4BPP	(4)

#define _MIN(X,Y) (((X)<(Y))? (X):(Y))
#define _MAX(X,Y) (((X)>(Y))? (X):(Y))

#define WRAP_COORD(Val, Size) ((Val) & ((Size)-1))

#define CLAMP(X, lower, upper) (_MIN(_MAX((X),(lower)), (upper)))

#define POWER_OF_2(X)   util_number_is_power_2(X)

//
// Define an expression to either wrap or clamp large or small vals to the
// legal coordinate range
//
#define LIMIT_COORD(Val, Size, AssumeImageTiles) \
      ((AssumeImageTiles)? WRAP_COORD((Val), (Size)): CLAMP((Val), 0, (Size)-1))
	  
#define RGBA(r, g, b, a)   ((GLuint) (((a) << 24) | ((b) << 16) | ((g) << 8) | (r)))

/****************************************************************************
** Defines
****************************************************************************/

#ifdef DEBUG
#include <assert.h>
#define _CRT_WARN 0
#define _RPT0(a,b) printf(b)
#define _RPT1(a,b,c) printf(b, c)
#define _RPT2(a,b,c,d) printf(b,c,d)
#define _RPT3(a,b,c,d,e) printf(b,c,d,e)
#define _RPT4(a,b,c,d,e,f) printf(b,c,d,f)
#ifndef _ASSERT
#define _ASSERT(X) assert(X)
#endif
#else
#define _CRT_WARN 0
#define _RPT0(a,b) printf(b)
#define _RPT1(a,b,c) printf(b, c)
#define _RPT2(a,b,c,d) printf(b,c,d)
#define _RPT3(a,b,c,d,e) printf(b,c,d,e)
#define _RPT4(a,b,c,d,e,f) printf(b,c,d,f)

#define _ASSERT(X) //
#endif

// Window Width and Height will be set in DisplayText depend of Screen orientation.
extern float WindowHeight; 
extern float WindowWidth;


#endif // MACROS_H_
