#ifndef OPENTISSUE_CORE_MATH_MATH_EULER_ANGLES_H
#define OPENTISSUE_CORE_MATH_MATH_EULER_ANGLES_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

#include <OpenTissue/core/math/math_quaternion.h> // Needed for quaternion

#include <OpenTissue/core/math/math_functions.h> // Needed for clamp
#include <OpenTissue/core/math/math_is_number.h> // Needed for is_number


#include <cmath>  // Needed for cos, atan2, and asin
#include <cassert>


namespace OpenTissue
{

  namespace math
  {

    /**
    * Computes Euler Angles.
    * The resulting order is: xyz, i.e. first x, then y and finally z
    *
    *
    * @param R    A rotation matrix
    *
    * @param rx   Upon return holds the rotation around x-axis in radians.
    * @param ry   Upon return holds the rotation around y-axis in radians.
    * @param rz   Upon return holds the rotation around z-axis in radians.
    *
    * @return     If return value is true then we have computed unique euler angle
    *             values otherwise our solution might not be unique
    */
    template<typename matrix3x3_type,typename real_type>
    inline bool euler_angles(matrix3x3_type const & R,real_type & rx, real_type & ry, real_type & rz)
    {
      using std::atan2;
      using std::asin;

      typedef typename matrix3x3_type::value_type    T;
      typedef typename matrix3x3_type::value_traits  value_traits;

      T const & m00 = R(0,0);
      T const & m01 = R(0,1);
      T const & m02 = R(0,2);
      T const & m10 = R(1,0);
      T const & m20 = R(2,0);
      T const & m21 = R(2,1);
      T const & m22 = R(2,2);
      // rot =  cy*cz           cz*sx*sy-cx*sz  cx*cz*sy+sx*sz
      //        cy*sz           cx*cz+sx*sy*sz -cz*sx+cx*sy*sz
      //       -sy              cy*sx           cx*cy
      if ( m20 < value_traits::one() )
      {
        if ( m20 > -value_traits::one() )
        {
          rz = boost::numeric_cast<real_type>( atan2(m10,m00) );
          ry = boost::numeric_cast<real_type>( asin(-m20)     );
          rx = boost::numeric_cast<real_type>( atan2(m21,m22) );
          return true;
        }

        // WARNING.  Not unique.  ZA - XA = -atan2(r01,r02)
        rz = boost::numeric_cast<real_type>( - atan2(m01,m02) );
        ry = value_traits::pi_2();
        rx = value_traits::zero();
        return false;
      }

      // WARNING.  Not unique.  ZA + XA = atan2(-r01,-r02)
      rz = boost::numeric_cast<real_type>( atan2(-m01,-m02) );
      ry = value_traits::pi_2();
      rx = value_traits::zero();
      return false;
    }










    /*
    * Computes ZYZ Euler Angles.
    *
    * @param Q    A quaternion representing some given orientation for which we wish to find the corresponding ZYZ Euler parameters.
    *
    * @param phi    The first rotation around the Z-axis.
    * @param psi    The rotation around the Y-axis.
    * @param theta  The first rotation around the Z-axis.
    *
    * @return     If return value is true then we have computed unique euler angle
    *             values otherwise our solution might not be unique
    */
    template<typename T>
    inline void ZYZ_euler_angles( OpenTissue::math::Quaternion<T> const & Q, T & phi, T & psi, T & theta  )
    {
      using std::atan2;
      using std::cos;

      typedef typename OpenTissue::math::Quaternion<T>  quaternion_type;
      typedef typename quaternion_type::vector3_type    vector3_type;
      typedef typename quaternion_type::value_traits    value_traits;

      phi   = value_traits::zero();
      psi   = value_traits::zero();
      theta = value_traits::zero();

      // Here phi, psi and theta defines the relative rotation, Q, such that
      //
      // Q ~ Rz( phi )*Ry( psi )*Rz(theta);
      //
      // Our taks is to find phi, psi, and theta given Q
      //
      // We exploit the following idea below to reduce the problem. We
      // use a clever test-vector, k = [0 0 1]^T and try to rotate this
      // vector with the given rotation. That is
      //
      // Q k Q^*  = Rz( phi )*Ry( psi )*Rz(theta) k =  Rz( phi )*Ry( psi ) k;
      //
      // denoting Q k Q^* = u, a unit vector, we no longer need to worry about theta.
      vector3_type const k = vector3_type(value_traits::zero(),value_traits::zero(),value_traits::one());
      vector3_type const u = Q.rotate(k);
      // Now we must have
      //
      //  u = Rz(phi) Ry(psi) [0 0 1]^T
      //
      // From this we must have
      //
      //  | u_x |   |  cos(phi) -sin(phi) 0 | |  cos(psi) 0 sin(psi) |   |0|
      //  | u_y | = |  sin(phi)  cos(phi) 0 | |  0        1 0        |   |0|
      //  | u_z |   |  0         0        1 | | -sin(psi) 0 cos(psi) |   |1|
      //
      // Multipling once
      //
      //  | u_x |   |  cos(phi) -sin(phi) 0 | |  sin(psi) |  
      //  | u_y | = |  sin(phi)  cos(phi) 0 | |   0       | 
      //  | u_z |   |  0         0        1 | |  cos(psi) |  
      //
      // Multipling twice
      //
      //  | u_x |   |  cos(phi)  sin(psi) |  
      //  | u_y | = |  sin(phi)  sin(psi) | 
      //  | u_z |   |  cos(psi)           |  
      //
      // From the third equation we solve
      //
      //  psi = acos( u_z )
      //
      // This forces psi to always be in the internval [0..pi].
      //
      T const u_z = clamp( u(2), -value_traits::one(), value_traits::one());
      assert(is_number(u_z) || !"ZYZ_euler_angles(): not an number encountered");
      assert(u_z <= value_traits::one() || !"ZYZ_euler_angles(): u_z was too big");
      assert(u_z >= -value_traits::one() || !"ZYZ_euler_angles(): u_z was too small");

      psi = boost::numeric_cast<T>( acos(u_z)   );
      assert(is_number(psi) || !"ZYZ_euler_angles(): psi was not an number encountered");
      assert(psi <= value_traits::pi() || !"ZYZ_euler_angles(): psi was too big");
      assert(psi >= value_traits::zero() || !"ZYZ_euler_angles(): psi was too small");
      //
      // We know that sin(psi) is allways going to be positive, which mean
      // that we can divide the second equation by the first equation and
      // obtain
      //
      //  sin(phi)/cos(phi) = tg(phi) = u_y/u_x
      //
      // From this we have
      //
      //  phi = arctan2( u_y, u_x )
      //
      // That means that phi will always be in the interval [-pi..pi].
      //
      // Observe that if psi is zero then u_y and u_x is both zero and our
      // approach will alway compute phi to be the value zero. The case
      // is actually worse than it seems. Because with psi is zero ZYZ are
      // in a gimbal lock where the two Z-axis transformations are completely
      // aligned.
      //
      //
      T const too_small = boost::numeric_cast<T>( 0.0001 );
      if(psi<too_small)
      {
        //
        // Our solution is to use another clever test vector 
        //
        vector3_type const i = vector3_type(value_traits::one(),value_traits::zero(),value_traits::zero());
        vector3_type const w = Q.rotate(i);

        T const w_x = w(0);
        T const w_y = w(1);
        assert(is_number(w_x) || !"ZYZ_euler_angles(): w_x was not an number encountered");
        assert(is_number(w_y) || !"ZYZ_euler_angles(): w_y not an number encountered");
        phi = boost::numeric_cast<T>( atan2(w_y,w_x) );
        assert(is_number(phi) || !"ZYZ_euler_angles(): phi was not an number encountered");
        assert(phi <=  value_traits::pi() || !"ZYZ_euler_angles(): phi was too big");
        assert(phi >= -value_traits::pi() || !"ZYZ_euler_angles(): phi was too small");

        //
        // psi was too close to zero so we are in a gimbal lock, we simply keep theta zero
        //
        return;
      }
      else
      {
        // We are not close to gimbal lock, so we can safely
        T const u_x = u(0);
        T const u_y = u(1);
        assert(is_number(u_x) || !"ZYZ_euler_angles(): u_x was not an number encountered");
        assert(is_number(u_y) || !"ZYZ_euler_angles(): u_y not an number encountered");
        phi = boost::numeric_cast<T>( atan2(u_y,u_x) );
        assert(is_number(phi) || !"ZYZ_euler_angles(): phi was not an number encountered");
        assert(phi <=  value_traits::pi() || !"ZYZ_euler_angles(): phi was too big");
        assert(phi >= -value_traits::pi() || !"ZYZ_euler_angles(): phi was too small");
      }


      //
      // So now we have 
      //
      //   Qzy =~ Rz( phi )*Ry( psi );
      //
      // and from this we know
      //
      //   Q = Qzy Qz(theta);
      //
      // so
      //
      //  (Qzy^* Q) = Qz(theta)
      //
      // We also know
      //
      //  (Qzy^* Q) = [s,v] = [ cos(theta/2) , sin(theta/2) * k ]
      //
      // where s is a value_type and v is a 3-vector. k is a unit length z-axis and
      // theta is a rotation along that axis.
      //
      // we can get theta/2 by:
      //
      //   theta/2 = atan2 ( sin(theta/2) , cos(theta/2) )
      //
      // but we can not get sin(theta/2) directly, only its absolute value:
      //
      //   |v| = |sin(theta/2)| * |k|  = |sin(theta/2)|
      //
      quaternion_type Qy;
      quaternion_type Qz;
      quaternion_type H;
      Qy.Ry(psi);
      Qz.Rz(phi);
      H = prod( conj( prod( Qz , Qy ) ), Q );

      vector3_type const i = vector3_type(value_traits::one(),value_traits::zero(),value_traits::zero());
      vector3_type const w = H.rotate(i);

      T const w_x = w(0);
      T const w_y = w(1);
      assert(is_number(w_x) || !"ZYZ_euler_angles(): w_x was not an number encountered");
      assert(is_number(w_y) || !"ZYZ_euler_angles(): w_y not an number encountered");
      theta = boost::numeric_cast<T>( atan2(w_y,w_x) );
      assert(is_number(theta) || !"ZYZ_euler_angles(): phi was not an number encountered");
      assert(theta <=  value_traits::pi() || !"ZYZ_euler_angles(): phi was too big");
      assert(theta >= -value_traits::pi() || !"ZYZ_euler_angles(): phi was too small");



      //T ct2 = Q.s();       //---   cos(theta/2)
      //T st2 = length( v ); //---  |sin(theta/2)|

      //// First try positive choice of sin(theta/2)
      //theta = value_traits::two()* atan2(st2,ct2);


      return;
    }

  } // namespace math

} // namespace OpenTissue

//OPENTISSUE_CORE_MATH_MATH_EULER_ANGLES_H
#endif
