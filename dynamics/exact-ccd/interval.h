// ---------------------------------------------------------
//
//  interval.h
//  Tyson Brochu 2011
//
// ---------------------------------------------------------

#ifndef TUNICATE_INTERVAL_H
#define TUNICATE_INTERVAL_H

#include <cassert>
#ifdef _WIN32
#include <float.h>
#pragma float_control(precise, on)
#pragma fenv_access(on)
//#pragma float_control(except, on)
// The following line is needed on Itanium processors.
#pragma fp_contract(off)

#else
#include <fenv.h>
#endif

#include "interval_base.h"

class Interval;
typedef Interval IntervalType;

#ifndef DEBUG
#define VERIFY() (void)0;
#else
#define VERIFY() assert( -v[0] <= v[1] ); \
assert( v[0] == v[0] ); \
assert( v[1] == v[1] );
#endif

#ifndef DEBUG
#define CHECK_ROUNDING_MODE() (void)0;
#else
#define CHECK_ROUNDING_MODE() assert( fegetround( ) == FE_UPWARD ); 
#endif


// ----------------------------------------
//
// class Interval:
//
// Stores the interval [a,b] as [-a,b] internally.  With proper arithmetic operations, this 
// allows us to use only FE_UPWARD and avoid switching rounding modes over and over.
//
// ----------------------------------------


class Interval : public IntervalBase
{
#ifdef _WIN32
    static unsigned int s_previous_rounding_mode;
#else
    static int s_previous_rounding_mode;
#endif
public:

   // Internal representation
   double v[2];
         
   Interval( double val );   
   Interval( double left, double right );
   Interval();
   
   virtual ~Interval() {}
   
   virtual double stored_left() const;
   virtual double stored_right() const;
   
   virtual Interval& operator+=(const Interval &rhs);
   virtual Interval& operator-=(const Interval &rhs);
   virtual Interval& operator*=(const Interval &rhs);
   
   virtual Interval operator+(const Interval &other) const;
   virtual Interval operator-(const Interval &other) const;
   virtual Interval operator*(const Interval &other) const;
   
   virtual Interval operator-( ) const;
   
   static void begin_special_arithmetic();
   static void end_special_arithmetic();
   
};

inline void create_from_double( double a, Interval& out );


// ----------------------------------------

inline Interval::Interval( double val )
{
   v[0] = -val;
   v[1] = val;
   VERIFY();
}

// ----------------------------------------

inline Interval::Interval( double left, double right )
{
   assert( left <= right );
   v[0] = -left;
   v[1] = right;
   VERIFY();
}

// ----------------------------------------

inline Interval::Interval()
{
   v[0] = 0;
   v[1] = 0;
   VERIFY();
}

// ----------------------------------------

inline double Interval::stored_left() const
{
   return v[0];
}

// ----------------------------------------

inline double Interval::stored_right() const
{
   return v[1];
}


// ----------------------------------------

inline Interval& Interval::operator+=(const Interval &rhs)
{
   CHECK_ROUNDING_MODE();
   VERIFY();
   v[0] += rhs.v[0];
   v[1] += rhs.v[1];
   VERIFY();
   
   return *this;
}

// ----------------------------------------

inline Interval& Interval::operator-=( const Interval& rhs )
{
   CHECK_ROUNDING_MODE();   
   v[0] += rhs.v[1];
   v[1] += rhs.v[0];
   VERIFY();
   return *this;
}

// ----------------------------------------

inline Interval& Interval::operator*=( const Interval& rhs )
{
   CHECK_ROUNDING_MODE();   
   Interval p = (*this) * rhs;
   *this = p;
   return *this;
}

// ----------------------------------------

inline Interval Interval::operator+(const Interval &other) const 
{
   CHECK_ROUNDING_MODE();   
   double v0 = v[0] + other.v[0];
   double v1 = v[1] + other.v[1  ];
   return Interval(-v0, v1);
}

// ----------------------------------------

inline Interval Interval::operator-(const Interval &other) const 
{
   CHECK_ROUNDING_MODE();   
   double v0 = v[0] + other.v[1];
   double v1 = v[1] + other.v[0];
   return Interval(-v0, v1);              
}

// ----------------------------------------

inline Interval Interval::operator*(const Interval &other) const
{
   CHECK_ROUNDING_MODE();
   
   double neg_a = v[0];
   double b = v[1];
   double neg_c = other.v[0];
   double d = other.v[1];
   
   Interval product;
   
   if ( b <= 0 )
   {
      if ( d <= 0 )
      {
         product.v[0] = -b * d;
         product.v[1] = neg_a * neg_c;
      }
      else if ( -neg_c <= 0 && 0 <= d )
      {
         product.v[0] = neg_a * d;
         product.v[1] = neg_a * neg_c;
      }
      else
      {
         product.v[0] = neg_a * d;
         product.v[1] = b * -neg_c;
      }
   }
   else if ( -neg_a <= 0 && 0 <= b )
   {
      if ( d <= 0 )
      {
         product.v[0] = b * neg_c;
         product.v[1] = neg_a * neg_c;
      }
      else if ( -neg_c <= 0 && 0 <= d )
      {
         product.v[0] = std::max( neg_a * d, b * neg_c );
         product.v[1] = std::max( neg_a * neg_c, b * d );
      }
      else
      {
         product.v[0] = neg_a * d;
         product.v[1] = b * d;
      }
      
   }
   else
   {
      if ( d <= 0 )
      {
         product.v[0] = b * neg_c; 
         product.v[1] = -neg_a * d;
      }
      else if ( -neg_c <= 0 && 0 <= d )
      {
         product.v[0] = b * neg_c;
         product.v[1] = b * d;
      }
      else
      {
         product.v[0] = -neg_a * neg_c;
         product.v[1] = b * d;
      }
   }
   
   return product;
   
}


// ----------------------------------------

inline Interval Interval::operator-( ) const
{
   CHECK_ROUNDING_MODE();
   return Interval( -v[1], v[0] );   
}

// ----------------------------------------

inline void Interval::begin_special_arithmetic()
{
#ifdef _WIN32
	s_previous_rounding_mode = _controlfp(0,0);
	_controlfp(_RC_UP,_MCW_RC);	
#else
	s_previous_rounding_mode = fegetround();
   fesetround( FE_UPWARD );
#endif
}

// ----------------------------------------

inline void Interval::end_special_arithmetic()
{
#ifdef _WIN32
	_controlfp(s_previous_rounding_mode,_MCW_RC);
#else
   fesetround( s_previous_rounding_mode );
#endif
}


// ----------------------------------------

inline void create_from_double( double a, Interval& out )
{
   out.v[0] = -a;
   out.v[1] = a;   
}


#endif

