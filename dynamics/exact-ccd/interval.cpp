// ---------------------------------------------------------
//
//  interval.cpp
//  Tyson Brochu 2011
//
// ---------------------------------------------------------

#include "interval.h"

#ifdef _WIN32
unsigned int Interval::s_previous_rounding_mode = ~0;
#else
int Interval::s_previous_rounding_mode = ~0;
#endif


