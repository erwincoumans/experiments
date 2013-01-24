
#ifndef MEMORYMGR_H_
#define MEMORYMGR_H_

#include "Macros.h"

#include <cstddef>
#include <algorithm>
#include <iostream>
#include <cstddef>

using namespace std;

template <typename T>
T *reallocEM(T *array, size_t old_size, size_t new_size)
{
   T *temp = new T[new_size];

   delete [] array;
   
#ifdef DEBUG
	char strData[128];
  	sprintf(strData,"%5s %3d %5s %3d", 
            "New memory piece", (int)new_size, "Old memory piece", (int)old_size);
#endif

   return copy(array, array + old_size, temp);
}


template <typename T>
bool SafeAlloc(T* &ptr, size_t cnt)
{
	_ASSERT(!ptr);
	if(cnt)
	{
		ptr = new T[cnt * sizeof(T)];
		_ASSERT(ptr);
		if(!ptr)
			return false;
	}
#ifdef DEBUG
	char strData[128];
  	sprintf(strData,"%5s %3d", 
            "Memory", (int)cnt);
//	LOG(string(strData), Logger::LOG_BLOK);
#endif
	
	memset(ptr, 0, cnt *sizeof(T));

	return true;
}

template <typename T>
void SafeRealloc(T* &ptr, size_t cnt)
{
   // simulate what realloc would do
   // allocate enough so that the old and the new memory have space
   T *temp = new T[cnt * sizeof(T)];
   
   size_t old_size = sizeof(*ptr);
   
   // delete old memory
   delete [] ptr;
   
   // copy temp starting whereever ptr has started before
   copy(ptr, ptr + old_size, temp);

#ifdef DEBUG
	char strData[128];
  	sprintf(strData,"%5s %3d %5s %3d", 
            "New memory piece", (int)cnt, "Old memory piece", (int)old_size);
//	LOG(string(strData), Logger::LOG_BLOK);
#endif
 	_ASSERT(ptr);
}

char* StrDup(const char *string);
#endif
