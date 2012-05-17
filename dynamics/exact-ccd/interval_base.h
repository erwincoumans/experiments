
// ---------------------------------------------------------
//
//  interval_base.h
//  Tyson Brochu 2011
//
//  Interface definition for Interval types.
//
// ---------------------------------------------------------

#ifndef TUNICATE_INTERVAL_BASE_H
#define TUNICATE_INTERVAL_BASE_H

#include "vec.h"

class IntervalBase
{
   
public:
   
   inline virtual ~IntervalBase() {}

   virtual double stored_left() const = 0;
   virtual double stored_right() const = 0;

   inline virtual Vec2d get_actual_interval() const;
   inline virtual Vec2d get_internal_representation() const;
      
   inline virtual bool contains_zero() const;
   inline virtual bool indefinite_sign() const;
   inline virtual bool is_certainly_negative( ) const;   
   inline virtual bool is_certainly_positive( ) const;
   inline virtual bool is_certainly_zero( ) const;
   
   inline virtual bool certainly_opposite_sign( const IntervalBase& other ) const;
   
   inline virtual bool same_sign( const IntervalBase& other ) const;
   
   inline virtual double estimate() const;
      
};


// ----------------------------------------

inline Vec2d IntervalBase::get_actual_interval() const
{
   return Vec2d( -stored_left(), stored_right() );
}

// ----------------------------------------

inline Vec2d IntervalBase::get_internal_representation() const
{
   return Vec2d( stored_left(), stored_right() );
}


// ----------------------------------------

inline bool IntervalBase::contains_zero() const
{
   // true if a <= 0 && b >= 0
   // remember v[0] == -a
   
   return ( stored_left() >= 0 && stored_right() >= 0 );
}

// ----------------------------------------
// TODO: This is equivalent to contains_zero(), should be removed

inline bool IntervalBase::indefinite_sign() const
{
   return ( stored_left() >= 0 && stored_right() >= 0 );
}

// ----------------------------------------

inline bool IntervalBase::is_certainly_negative( ) const
{
   return ( stored_right() < 0 );
}

// ----------------------------------------

inline bool IntervalBase::is_certainly_positive( ) const
{
   return ( stored_left() < 0 );
}

// ----------------------------------------

inline bool IntervalBase::is_certainly_zero( ) const
{
   return ( stored_left() == 0 && stored_right() == 0 );
}

// ----------------------------------------

inline bool IntervalBase::certainly_opposite_sign( const IntervalBase& other ) const
{
   return ( ( this->is_certainly_negative() && other.is_certainly_positive() ) ||
           ( this->is_certainly_positive() && other.is_certainly_negative() ) );     
}

// ----------------------------------------

inline bool IntervalBase::same_sign( const IntervalBase& other ) const
{
   return ( this->contains_zero() ||
           other.contains_zero() ||
           ( this->is_certainly_negative() && other.is_certainly_negative() ) || 
           ( this->is_certainly_positive() && other.is_certainly_positive() ) );
}


// ----------------------------------------

inline double IntervalBase::estimate() const
{
   double est = 0.5 * (-stored_left() + stored_right());
   
   if ( est == 0 && !is_certainly_zero() )
   {
      // don't return zero as an estimate if the interval is not identically zero.
      est = 1e-20;
   }
   
   return est;
}

// ----------------------------------------

inline bool certainly_opposite_sign( const IntervalBase& a, const IntervalBase& b ) 
{
   return a.certainly_opposite_sign( b );
}

// ----------------------------------------

inline bool same_sign( const IntervalBase& a, const IntervalBase& b )
{  
   return a.same_sign(b);
}


#endif

