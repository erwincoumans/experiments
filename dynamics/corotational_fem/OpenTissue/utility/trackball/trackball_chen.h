#ifndef OPENTISSUE_UTILITY_TRACKBALL_CHEN_H
#define OPENTISSUE_UTILITY_TRACKBALL_CHEN_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

#include <OpenTissue/utility/trackball/trackball_generic.h>
#include <OpenTissue/core/math/math_constants.h>

namespace OpenTissue
{
  namespace utility
  {
    namespace trackball
    {

      template<typename real_type_ = double>
      class Chen : public GenericTrackball<real_type_>
      {
      public:

        typedef GenericTrackball<real_type_>             base_type;
        typedef typename base_type::real_type            real_type;
        typedef typename base_type::vector3_type         vector3_type;
        typedef typename base_type::matrix3x3_type       matrix3x3_type;
        typedef typename base_type::quaternion_type      quaternion_type;
        typedef typename base_type::gl_transform_type    gl_transform_type;

      public:

        Chen() 
          : base_type()
        {
          reset();
        }

        Chen(real_type const & radius) 
          : base_type(radius)
        {
          reset();
        }

        void reset()    {      base_type::reset();    }

        void begin_drag(real_type const & x, real_type const & y)
        {
          this->m_xform_anchor = this->m_xform_current;
          this->m_xform_incremental = diag(1.0);
          this->m_xform_current = diag(1.0);
          this->m_anchor_position = vector3_type(x,y,0);
          this->m_current_position = vector3_type(x,y,0);
        }

        void drag(real_type const & x, real_type const & y)
        {
          this->m_current_position = vector3_type(x, y, 0);
          compute_incremental(this->m_anchor_position,this->m_current_position,this->m_xform_incremental);
        }

        void end_drag(real_type const & x, real_type const & y)
        {
          this->m_current_position = vector3_type(x, y, 0);
          compute_incremental(this->m_anchor_position,this->m_current_position,this->m_xform_incremental);
        }

      private:

        real_type f(real_type const & x) const
        {
          if (x <= 0) return 0;
          if (x >= 1) return math::detail::pi_2<real_type>();
          return math::detail::pi_2<real_type>() * x;
        }

        void compute_incremental(vector3_type const & anchor, vector3_type const & current, matrix3x3_type & transform)
        {
          using std::cos;
          using std::sin;
          using std::fabs;
          real_type length_anchor = length(anchor);
          vector3_type Pa = unit(anchor);
          vector3_type Pc = unit(current);
          vector3_type aXc = Pa % Pc;
          real_type tau   = atan2(length(aXc), Pa * Pc ); 
          real_type phi   = atan2(anchor(1), anchor(0));
          real_type omega = f(length_anchor / this->m_radius);
          this->m_axis = unit(
            vector3_type(
            -cos(tau) * sin(phi) - cos(omega) * cos(phi) * sin(tau)
            , cos(phi) * cos(tau) - cos(omega) * sin(phi) * sin(tau)
            , sin(omega) * sin(tau)
            )
            );

          vector3_type d   = current - anchor;
          this->m_angle = math::detail::pi_2<real_type>() * length(d) / this->m_radius * (1 - (1 - 0.2 / math::detail::pi<real_type>()) * 2 * omega / math::detail::pi<real_type>() *(1 - fabs(cos(tau))));
          transform = Ru(this->m_angle,this->m_axis);
        }
      };
    } // namespace trackball
  } // namespace utility
} // namespace OpenTissue

// OPENTISSUE_UTILITY_TRACKBALL_CHEN_H
#endif
