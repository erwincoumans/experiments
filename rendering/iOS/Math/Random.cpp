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

#include "Random.h"
#include <stdlib.h>

// global default random number generator
CTrivialRandomGenerator gRandom(12345);

// TODO: use something better than rand(), such as a twister or r250
void CTrivialRandomGenerator::setSeed(U32 seed)
{
   mSeed = seed;
}

// we use rand_r here to keep various instances independant
S32 CTrivialRandomGenerator::randI()
{
   S32 ret = rand_r(&mSeed);
   return ret;
}