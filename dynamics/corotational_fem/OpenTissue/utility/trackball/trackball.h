#ifndef OPENTISSUE_UTILITY_TRACKBALL_TRACKBALL_H
#define OPENTISSUE_UTILITY_TRACKBALL_TRACKBALL_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

#include <OpenTissue/utility/trackball/trackball_generic.h>

namespace OpenTissue
{
  namespace utility
  {
    namespace trackball
    {

      template<typename real_type_ = double>
      class Trackball : public GenericTrackball<real_type_>
      {
      public:

        typedef GenericTrackball<real_type_>             base_type;
        typedef typename base_type::real_type            real_type;
        typedef typename base_type::vector3_type         vector3_type;
        typedef typename base_type::matrix3x3_type       matrix3x3_type;
        typedef typename base_type::quaternion_type      quaternion_type;
        typedef typename base_type::gl_transform_type    gl_transform_type;

      public:

        Trackball() 
          : base_type()
        {
          reset();
        }

        Trackball(real_type const & radius) 
          : base_type(radius)
        {
          reset();
        }

        void reset()
        {
          base_type::reset();
          project_onto_surface(this->m_anchor_position);
          project_onto_surface(this->m_current_position);
        }

        void begin_drag(real_type const & x, real_type const & y)
        {
          this->m_angle = 0.0;
          this->m_axis.clear();

          this->m_xform_anchor = this->m_xform_current;
          this->m_xform_incremental = math::diag<real_type>( 1.0 );
          this->m_xform_current = math::diag<real_type>( 1.0 );

          this->m_anchor_position = vector3_type(x,y,0);
          project_onto_surface(this->m_anchor_position);
          this->m_current_position = vector3_type(x,y,0);
          project_onto_surface(this->m_current_position);
        }

        void drag(real_type const & x, real_type const & y)
        {
          this->m_current_position = vector3_type(x,y,0);
          project_onto_surface(this->m_current_position);
          compute_incremental(this->m_anchor_position,this->m_current_position,this->m_xform_incremental);
        }

        void end_drag(real_type const & x, real_type const & y)
        {
          this->m_current_position = vector3_type(x,y,0);
          project_onto_surface(this->m_current_position);
          compute_incremental(this->m_anchor_position,this->m_current_position,this->m_xform_incremental);
        }

      private:

        void project_onto_surface(vector3_type & P)
        {
          using std::sqrt;
          const static real_type radius2 = this->m_radius * this->m_radius;
          real_type length2 = P(0)*P(0) + P(1)*P(1);
          if (length2 <= radius2 / 2.0)
            P(2) = sqrt(radius2 - length2);
          else
          {
            P(2) = radius2 / (2.0 * sqrt(length2));
            real_type length = sqrt(length2 + P[2] * P[2]);
            P /= length;
          }
          P = unit(P);
        }

        void compute_incremental(vector3_type const & anchor, vector3_type const & current, matrix3x3_type & transform)
        {
          vector3_type axis = anchor % current;
          this->m_axis = unit(axis);
          this->m_angle = atan2(length(axis), anchor * current );
          transform = Ru(this->m_angle,this->m_axis);
        }

      };
    } // namespace trackball
  } // namespace utility
} // namespace OpenTissue

// OPENTISSUE_UTILITY_TRACKBALL_BELL_H
#endif
