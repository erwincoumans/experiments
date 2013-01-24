/*
Oolong Engine for the iPhone / iPod touch
Copyright (c) 2008 Paul Scott et al http://code.google.com/p/oolongengine/

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

#ifndef _OOLONG_TYPES_H_
#define _OOLONG_TYPES_H_

typedef signed char     S8;
typedef unsigned char   U8;

typedef signed short    S16;
typedef unsigned short  U16;

typedef signed int      S32;
typedef unsigned int    U32;

typedef float           F32;
typedef double          F64;

typedef signed long long    S64;
typedef unsigned long long  U64;


// check that the types are the correct sizes via some template magic
#if defined(DEBUG)
namespace oolongengine_types_tests
{

typedef struct EmptyType { };

template <bool Flag, typename IfTrue, typename IfFalse> struct if_c;

template <typename IfTrue, typename IfFalse>
struct if_c <true, IfTrue, IfFalse> { typedef typename IfTrue::type type; };
 
template <typename IfTrue, typename IfFalse>
struct if_c <false, IfTrue, IfFalse> { typedef typename IfFalse::type type; };

struct assert_pass { typedef bool type; };

template<typename BadType, typename T, int goodsize, int badsize>
struct assert_fail { typedef typename T::error_Assert_Failed type; };

// assert type sizes
#define ASSERT_SIZEOF( T, S ) \
typedef if_c< sizeof( T ) == S / 8, assert_pass, assert_fail<T, EmptyType, sizeof( T ), S / 8 > >::type assert_sizeof_##T;

ASSERT_SIZEOF( S8,   8 )
ASSERT_SIZEOF( U8,   8 )

ASSERT_SIZEOF( S16,  16 )
ASSERT_SIZEOF( U16,  16 )

ASSERT_SIZEOF( S32,  32 )
ASSERT_SIZEOF( U32,  32 )

ASSERT_SIZEOF( S64,  64 )
ASSERT_SIZEOF( U64,  64 )

ASSERT_SIZEOF( F32, 32 )
ASSERT_SIZEOF( F64, 64 )

}
#endif // if defined(DEBUG)

#endif // guard