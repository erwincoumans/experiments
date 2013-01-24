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

#include "../Utility/Types.h"

class CRandomGenerator
{
protected:
   static const U32 smRandMax = 0x7fffffff;
public:
   virtual ~CRandomGenerator() { } 
   /// seed the random number generator
   virtual void setSeed(U32 seed) = 0;
   /// generates a random integer on the range [0, 2^31)
   virtual S32 randI() = 0;
   /// generate a random F32 number on the range [0.0 to 1.0]
   virtual F32 randF();
   S32 randI(S32 a, S32 b);
   F32 randF(F32 a, F32 b);
};

class CTrivialRandomGenerator : public CRandomGenerator
{
protected:
   U32 mSeed;
public:
   CTrivialRandomGenerator(U32 seed = 0) { setSeed(seed); };
   virtual void setSeed(U32 seed);
   virtual S32 randI();
};

inline F32 CRandomGenerator::randF()
{
   return F32(randI()) / F32(smRandMax);
}

inline S32 CRandomGenerator::randI(S32 a, S32 b)
{
   if(a > b) { a ^= b; b ^= a; a ^= b; }
   return a + (randI() % (b - a + 1));
}

inline F32 CRandomGenerator::randF(F32 a, F32 b)
{
   if(a > b) { F32 t = a; a = b; b = t; }
   return a + ((b - a) * randF());
}

extern CTrivialRandomGenerator gRandom;
