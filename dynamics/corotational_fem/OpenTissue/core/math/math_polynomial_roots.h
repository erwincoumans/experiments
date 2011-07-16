#ifndef OPENTISSUE_CORE_MATH_MATH_POLYNOMIAL_ROOTS_H
#define OPENTISSUE_CORE_MATH_MATH_POLYNOMIAL_ROOTS_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

#include <OpenTissue/core/math/math_precision.h>

#include <OpenTissue/core/math/math_constants.h>

#include <cmath>

namespace OpenTissue
{

  namespace math
  {

    /**
    * Solve roots of
    *
    *   c1 x + c0 = 0
    *
    * @param       c1     The coefficient of the first power term.
    * @param       c0     The coefficient of the zeroth power term.
    * @param    roots     Upon return this array contains the computed
    *                     roots.
    * @param    count     Upon return this parameter contains the number
    *                     of computed roots.
    * @return             If roots were computed then the return
    *                     value is true otherwise it is false.
    */
    template<typename real_type, typename real_array>
    inline bool compute_polynomial_roots(real_type const & c0, real_type const & c1, unsigned int & count, real_array & roots)
    {
      using std::fabs;

      static real_type const epsilon = math::working_precision<real_type>();

      if ( fabs(c1) >= epsilon )
      {
        roots[0] = -c0/c1;
        count = 1;
        return true;
      }
      count = 0;
      return false;
    }

    /**
    * Solve roots of
    *
    *   c2 x^2 + c1 x + c0 = 0
    *
    * @param       c2     The coefficient of the second power term.
    * @param       c1     The coefficient of the first power term.
    * @param       c0     The coefficient of the zeroth power term.
    * @param    roots     Upon return this array contains the computed
    *                     roots.
    * @param    count     Upon return this parameter contains the number
    *                     of computed roots.
    * @return             If roots were computed then the return
    *                     value is true otherwise it is false.
    */
    template<typename real_type, typename real_array>
    inline bool compute_polynomial_roots(real_type const & c0, real_type const & c1, real_type const & c2, unsigned int & count, real_array & roots)
    {
      using std::fabs;
      using std::sqrt;

      static real_type const epsilon = math::working_precision<real_type>();
      static real_type const four    = boost::numeric_cast<real_type>(4.0);
      static real_type const two     = detail::two<real_type>();
      static real_type const zero    = detail::zero<real_type>();

      if ( fabs(c2) <= epsilon )
        return compute_polynomial_roots(c0,c1,count,roots);

      real_type discr = c1*c1 - four*c0*c2;

      if ( fabs(discr) <= epsilon )
      {
        //discr = zero;
        roots[0] = -c1/(two*c2);
        count = 1;
        return true;
      }

      if ( discr < zero )
      {
        count = 0;
        return false;
      }

      assert( discr > zero || !"compute_polynomial_roots(): discriminant was expected to be positive");
      
      discr = sqrt(discr);
      roots[0] = (-c1 - discr)/(two*c2);
      roots[1] = (-c1 + discr)/(two*c2);
      count = 2;
      return true;
    }

    /**
    * Solve roots of
    *
    *   c3 x^3 + c2 x^2 + c1 x + c0 = 0
    *
    * @param       c3     The coefficient of the third power term.
    * @param       c2     The coefficient of the second power term.
    * @param       c1     The coefficient of the first power term.
    * @param       c0     The coefficient of the zeroth power term.
    * @param    roots     Upon return this array contains the computed
    *                     roots.
    * @param    count     Upon return this parameter contains the number
    *                     of computed roots.
    * @return             If roots were computed then the return
    *                     value is true otherwise it is false.
    */
    template<typename real_type, typename real_array>
    inline bool compute_polynomial_roots(real_type c0, real_type c1, real_type c2, real_type const & c3, unsigned int & count, real_array & roots)
    {
      using std::sqrt;
      using std::fabs;
      using std::pow;
      using std::atan2;
      using std::cos;
      using std::sin;

      static real_type const THIRD         = boost::numeric_cast<real_type>( 1.0/3.0   );
      static real_type const TWENTYSEVENTH = boost::numeric_cast<real_type>( 1.0/27.0  );
      static real_type const NINE          = boost::numeric_cast<real_type>( 9.0       );
      static real_type const TWO           = detail::two<real_type>();
      static real_type const ZERO          = detail::zero<real_type>();
      static real_type const SQRT3         = boost::numeric_cast<real_type>( sqrt(3.0) );
      static real_type const epsilon       = math::working_precision<real_type>();

      if ( fabs(c3) <= epsilon )
      {
        return compute_polynomial_roots(c0,c1,c2,count,roots);
      }

      // make polynomial monic, x^3+c2*x^2+c1*x+c0
      c0 /= c3;
      c1 /= c3;
      c2 /= c3;

      // convert to y^3 + a*y + b = 0 by x = y - c2/3
      real_type offset = THIRD*c2;

      real_type A      = c1 - c2*offset;
      real_type B      = c0 + c2*(TWO*c2*c2 - NINE*c1)*TWENTYSEVENTH;
      real_type half_B = B/TWO;

      real_type discr = half_B*half_B + A*A*A*TWENTYSEVENTH;

      if ( fabs(discr) <= epsilon )
      {
        //discr = ZERO;

        //if ( half_B >= ZERO )
        //  temp = -pow(half_B,THIRD);
        //else
        //  temp = pow(-half_B,THIRD);
        real_type temp = ( half_B >= ZERO )? -pow(half_B,THIRD) : pow(-half_B,THIRD);

        roots[0] = TWO*temp - offset;
        roots[1] = -temp - offset;
        roots[2] = roots[1];
        count = 3;
      }
      else if ( discr > ZERO )  // 1 real, 2 complex roots
      {
        discr = sqrt(discr);
        
        real_type temp = -half_B + discr;
        //if ( temp >= ZERO )
        //  roots[0] = pow(temp,THIRD);
        //else
        //  roots[0] = -pow(-temp,THIRD);
        roots[0] = ( temp >= ZERO ) ? pow(temp,THIRD) : -pow(-temp,THIRD);
        
        temp = - half_B - discr;
        
        //if ( temp >= ZERO )
        //  roots[0] += pow(temp,THIRD);
        //else
        //  roots[0] -= pow(-temp,THIRD);
        roots[0] += ( temp >= ZERO ) ? pow(temp,THIRD) : -pow(-temp,THIRD);

        roots[0] -= offset;
        count = 1;
      }
      else if ( discr < ZERO )
      {
        real_type dist  = sqrt(-THIRD*A);
        real_type angle = THIRD*atan2(sqrt(-discr), -half_B);
        real_type c     = cos(angle);
        real_type s     = sin(angle);

        roots[0] = TWO*dist*c-offset;
        roots[1] = -dist*(c+SQRT3*s)-offset;
        roots[2] = -dist*(c-SQRT3*s)-offset;
        count = 3;
      }
      return true;
    }

    /**
    * Solve roots of
    *
    *  c4 x^4 +  c3 x^3 + c2 x^2 + c1 x + c0 = 0
    *
    * @param       c4     The coefficient of the fourth power term.
    * @param       c3     The coefficient of the third power term.
    * @param       c2     The coefficient of the second power term.
    * @param       c1     The coefficient of the first power term.
    * @param       c0     The coefficient of the zeroth power term.
    * @param    roots     Upon return this array contains the computed
    *                     roots.
    * @param    count     Upon return this parameter contains the number
    *                     of computed roots.
    * @return             If roots were computed then the return
    *                     value is true otherwise it is false.
    */
    template<typename real_type,typename real_array>
    inline bool compute_polynomial_roots(real_type c0, real_type c1, real_type c2, real_type c3, real_type const & c4, unsigned int & count, real_array & roots)
    {
      using std::fabs;
      using std::sqrt;

      static real_type const THREEQUATERS  = boost::numeric_cast<real_type>( 0.75 );
      static real_type const FOUR          = boost::numeric_cast<real_type>( 4.0 );
      static real_type const EIGHT         = boost::numeric_cast<real_type>( 8.0 );
      static real_type const ZERO          = detail::zero<real_type>();
      static real_type const TWO           = detail::two<real_type>();
      static real_type const ONE           = detail::one<real_type>();

      static real_type const epsilon = math::working_precision<real_type>();

      if ( fabs(c4) <= epsilon )
      {
        return compute_polynomial_roots(c0,c1,c2,c3,count,roots);
      }

      // make polynomial monic, x^4+c3*x^3+c2*x^2+c1*x+c0
      c0 /= c4;
      c1 /= c4;
      c2 /= c4;
      c3 /= c4;

      // reduction to resolvent cubic polynomial y^3+r2*y^2+r1*y+r0 = 0
      real_type r0 = -c3*c3*c0 + FOUR*c2*c0 - c1*c1;
      real_type r1 = c3*c1 - FOUR*c0;
      real_type r2 = -c2;
      compute_polynomial_roots(r0,r1,r2,ONE,count,roots);  // always produces at least one root
      real_type Y = roots[0];

      count = 0;
      real_type discr = (c3*c3)/FOUR - c2 + Y;

      if ( fabs(discr) <= epsilon )
        discr = ZERO;

      if ( discr < ZERO )
      {
        count = 0;
      }
      else if ( discr > ZERO )
      {
        real_type r       = sqrt(discr);
        real_type t1      = THREEQUATERS*c3*c3 - r*r - TWO*c2;
        real_type t2      = (FOUR*c3*c2 - EIGHT*c1 - c3*c3*c3) /(FOUR*r);

        real_type t_plus  = t1+t2;
        real_type t_minus = t1-t2;

        if ( fabs(t_plus) <= epsilon )
          t_plus = ZERO;
        if ( fabs(t_minus) <= epsilon )
          t_minus = ZERO;

        if ( t_plus >= ZERO )
        {
          real_type D = sqrt(t_plus);
          roots[0] = -c3/FOUR + (r+D)/TWO;
          roots[1] = -c3/FOUR + (r-D)/TWO;
          count += 2;
        }
        if ( t_minus >= ZERO )
        {
          real_type E = sqrt(t_minus);
          roots[count++] = -c3/FOUR + (E-r)/TWO;
          roots[count++] = -c3/FOUR - (E+r)/TWO;
        }
      }
      else
      {
        real_type t2 = Y*Y - FOUR*c0;
        if ( t2 >= - epsilon )
        {
          if ( t2 < ZERO ) // round to zero
            t2 = ZERO;

          t2 = TWO*sqrt(t2);
          real_type t1 = THREEQUATERS*c3*c3 - TWO*c2;
          if ( t1+t2 >= epsilon )
          {
            real_type D = sqrt(t1+t2);
            roots[0] = -c3/FOUR + D/TWO;
            roots[1] = -c3/FOUR - D/TWO;
            count += 2;
          }
          if ( t1-t2 >= epsilon )
          {
            real_type E = sqrt(t1-t2);
            roots[count++] = -c3/FOUR + E/TWO;
            roots[count++] = -c3/FOUR - E/TWO;
          }
        }
      }

      return count > 0;
    }

  } // namespace math

} // namespace OpenTissue

// OPENTISSUE_CORE_MATH_MATH_POLYNOMIAL_ROOTS_H
#endif
