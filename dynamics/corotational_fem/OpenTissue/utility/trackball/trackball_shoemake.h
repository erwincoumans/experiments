#ifndef OPENTISSUE_UTILITY_TRACKBALL_SHOEMAKE_H
#define OPENTISSUE_UTILITY_TRACKBALL_SHOEMAKE_H
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
      class Shoemake : public GenericTrackball<real_type_>
      {
      public:

        typedef GenericTrackball<real_type_>    base_type;
        typedef typename base_type::real_type            real_type;
        typedef typename base_type::vector3_type         vector3_type;
        typedef typename base_type::matrix3x3_type       matrix3x3_type;
        typedef typename base_type::quaternion_type      quaternion_type;
        typedef typename base_type::gl_transform_type    gl_transform_type;

      public:

        Shoemake() 
          : base_type()
        {
          reset();
        }

        Shoemake(real_type const & radius) 
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
          this->m_xform_incremental = diag(1.0);
          this->m_xform_current = diag(1.0);

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

          if (length2 <= radius2)
            P(2) = sqrt(radius2 - length2);
          else
          {
            P(0) *= this->m_radius / sqrt(length2);
            P(1) *= this->m_radius / sqrt(length2);
            P(2) = 0;
          }
        }

        void compute_incremental(vector3_type const & anchor, vector3_type const & current, matrix3x3_type & transform)
        {
          quaternion_type  Q_anchor ( 0, unit(anchor)  );
          quaternion_type  Q_current( 0, unit(current) );
          quaternion_type Q_rot = -prod(Q_current,Q_anchor);
          this->m_axis = Q_rot.v();
          this->m_angle = atan2(length(this->m_axis), Q_rot.s());
          transform = Q_rot;  //--- KE 20060307: Kaiip extracted axis and angle from this conversion!!!
          //--- I think there is a bug in this piece of code, m_axis is not guaranteed to be a unit-vector???
        }
      };

    } // namespace trackball
  } // namespace utility
} // namespace OpenTissue

// OPENTISSUE_UTILITY_TRACKBALL_SHOEMAKE_H
#endif
