//#define WIN32_LEAN_AND_MEAN
//#define WIN32_EXTRA_LEAN
//#include <windows.h>
//#include <assert.h>
#include "Timer.h"
#include <sys/time.h>

double Prof_get_time(void)
{
	timeval tv;
	gettimeofday(&tv,NULL);
	return (double)((tv.tv_sec*1000) + (tv.tv_usec/1000.0));
/*
   LARGE_INTEGER freq;
   LARGE_INTEGER time;

   BOOL ok = QueryPerformanceFrequency(&freq);
   assert(ok == TRUE);

   freq.QuadPart = freq.QuadPart;

   ok = QueryPerformanceCounter(&time);
   assert(ok == TRUE);

   return time.QuadPart / (double) freq.QuadPart;
*/
}
