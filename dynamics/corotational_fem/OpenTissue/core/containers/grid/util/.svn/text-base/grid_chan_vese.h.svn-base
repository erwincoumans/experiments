#ifndef OPENTISSUE_CORE_CONTAINERS_GRID_UTIL_CHANVESE_H
#define OPENTISSUE_CORE_CONTAINERS_GRID_UTIL_CHANVESE_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

#include <OpenTissue/core/math/math_vector3.h>
#include <OpenTissue/core/math/math_matrix3x3.h>
#include <OpenTissue/core/containers/grid/grid.h>

#include <boost/cast.hpp> //--- needed for numeric_cast

#include <iostream>
#include <cmath>
#include <algorithm>

namespace OpenTissue
{
  namespace grid
  {

    // TODO: henrikd 20050720 - We might have an issue regarding C_in/C_out being integral_types...

    namespace chan_vese{

      /**
      * Update Phi level set.
      * Applies the ``Chan-Vese'' speed-function to phi, result is returned in psi.
      *
      * @param phi      Input level set.
      * @param U        Input image.
      * @param C_in     Mean value of inside region. This is the mean value of the region that phi is supposed to enclose.
      * @param C_out    Mean value of outside region. This is the mean value of the region that is supposed to lie outside phi.
      * @param lambda   Weight of input/output regions (legal values 0..1) 0.5 means input and output are weighted equally.
      * @param mu       Mean Curvature regularization (requires input level set to be close to signed distance grid)
      * @param nu       Area (Well, it is really volume:-) regularization.
      * @param dt       Time-step to use in update.
      * @param psi      Output levelset. Note if input is a signed distance grid then output may not be a signed distance grid, thus you may need to redistance output levelset.
      *
      * @return         Maximum change in value of level set during update. This is usefull for steady state testing.
      */
      template<
        typename grid_type_in
        , typename image_grid_type
        , typename real_type
        , typename grid_type_out
      >
      inline real_type update(
        grid_type_in const & phi
      , image_grid_type const & U
      , typename image_grid_type::value_type const & C_in
      , typename image_grid_type::value_type const & C_out
      , real_type const & lambda
      , real_type const & mu
      , real_type const & nu
      , real_type const & dt
      , grid_type_out & psi
      )
      {
        using std::max;
        using std::fabs;

        typedef typename grid_type_in::value_type            input_type;
        typedef typename grid_type_out::value_type           output_type;
        typedef typename image_grid_type::value_type         image_type;
        typedef typename grid_type_in::const_index_iterator  input_iterator;
        typedef typename grid_type_out::index_iterator       output_iterator;

        assert(mu>=0     || !"update(): curvature coefficient must be non-negative");
        assert(nu>=0     || !"update(): area coefficient must be non-negative");
        assert(dt>0      || !"update(): time-step must be positive");
        assert(lambda>=0 || !"update(): region weight must be non-negative");
        assert(lambda<=1 || !"update(): region weight must be less than or equal to one");

        assert(phi.I() == psi.I() || !"update(): incompatible grid dimensions");
        assert(phi.J() == psi.J() || !"update(): incompatible grid dimensions");
        assert(phi.K() == psi.K() || !"update(): incompatible grid dimensions");

        real_type lambda_in   = lambda;
        real_type lambda_out  = real_type(1.0) - lambda;
        real_type max_delta   = real_type(0.0);//--- maximum change in level set update

        input_iterator input_begin = phi.begin();
        input_iterator input_end   = phi.end();
        input_iterator input       = input_begin;

        output_iterator output_begin = psi.begin();
        output_iterator output_end   = psi.end();
        output_iterator output       = output_begin;

        input_type unused = phi.unused();

        for(;input!=input_end;++input,++output)
        {
          if(*input!=unused)
          {
            input_type old_val = *input;

            image_type u0 = U(input.get_index());
            output_type speed  = output_type(0);
            if(mu)
            {
              input_type H = input_type(0);
              mean_curvature( input, H );
              speed -= mu*H;
            }
            speed += nu;
            speed -= lambda_in*(u0-C_in)*(u0-C_in);
            speed += lambda_out*(u0-C_out)*(u0-C_out);
            *output = old_val - dt*speed;

            max_delta = max(max_delta, fabs(*output - old_val) );
          }
        }
        return max_delta;
      }

      /**
      * Compute C_in
      *
      * @param phi    Level set.
      * @param U      An image.
      *
      * @return    The mean image value of U in the region inside
      *            the zero level set of phi.
      */
      template<typename grid_type,typename image_grid_type>
      inline typename image_grid_type::value_type compute_cin(
        grid_type const & phi
        , image_grid_type const & U
        )
      {
        typedef typename grid_type::value_type           input_type;
        typedef typename image_grid_type::value_type     image_type;
        typedef typename grid_type::const_index_iterator input_iterator;

        size_t voxels_inside  = 0;
        double C_in                 = 0.0;  //--- can not use image_type due to overflow problem
        input_iterator input_begin  = phi.begin();
        input_iterator input_end    = phi.end();
        input_iterator input;
        input_type unused = phi.unused();
        for(input=input_begin;input!=input_end;++input)
        {
          if(*input==unused)
            continue;
          if(*input<0)
          {
            C_in += U( input.get_index() );
            ++voxels_inside;
          }
        }
        C_in  /= voxels_inside;
        return image_type(C_in);
      }

      /**
      * Compute C_out
      *
      * @param phi    Level set.
      * @param U      An image.
      *
      * @return    The mean image value of U in the region
      *            outside the zero level set of phi.
      */
      template<typename grid_type,typename image_grid_type>
      inline typename image_grid_type::value_type compute_cout(
        grid_type const & phi
        , image_grid_type const & U
        )
      {
        typedef typename grid_type::value_type            input_type;
        typedef typename image_grid_type::value_type      image_type;
        typedef typename grid_type::const_index_iterator  input_iterator;

        size_t voxels_outside = 0;
        double   C_out              = 0.0; //--- can not use image_type due to overflow problem
        input_iterator input_begin  = phi.begin();
        input_iterator input_end    = phi.end();
        input_iterator input;
        input_type unused = phi.unused();
        for(input=input_begin;input!=input_end;++input)
        {
          if(*input==unused)
            continue;
          if(*input>=0)
          {
            C_out += U( input.get_index() );
            ++voxels_outside;
          }
        }
        C_out  /= voxels_outside;
        return image_type(C_out);
      }

      /**
      * Compute Volume.
      * Auxiliary method usefull for quick estimation of the volume
      * of a segmented region represented by the zero level set.
      *
      * @param phi    Level set grid.
      * @param V      Upon return contains the total volume of all
      *               voxels lying inside the zero levelset surface
      *               of phi.
      */
      template<typename grid_type,typename real_type>
      inline void compute_volume(
        grid_type const & phi
        , real_type & V
        )
      {
        typedef typename grid_type::const_iterator  input_iterator;
        typedef typename grid_type::value_type         input_type;

        size_t voxels  = 0;
        input_iterator input_begin = phi.begin();
        input_iterator input_end   = phi.end();
        input_iterator input;
        input_type unused = phi.unused();
        for(input=input_begin;input!=input_end;++input)
        {
          if(*input==unused)
            continue;
          if(*input<0)
            ++ voxels;
        }
        V = real_type( voxels*phi.dx()*phi.dy()*phi.dz());
      }

      /**
      * Initialize phi grid with threshold on image
      *
      * @param U         An image.
      * @param phi       Level set to be initialized.
      * @param min_value Minimum treshold value for inside region.
      * @param max_value Maximum treshold value for inside region.
      */
      template<typename grid_type,typename image_grid_type>
      inline void threshold_initialize(
        image_grid_type const & U
        , typename image_grid_type::value_type const& min_value
        , typename image_grid_type::value_type const& max_value
        , grid_type & phi
        )
      {
        using OpenTissue::grid::min_element;
        using OpenTissue::grid::max_element;

        typedef typename image_grid_type::value_type image_type;
        typedef typename grid_type::index_iterator   output_iterator;

        assert(min_value < max_value     || !"threshold_initialize(): min value must be less than max value");
        assert(min_value>min_element(U)  || !"threshold_initialize(): min value must be greather than minimum image value");
        assert(max_value<max_element(U)  || !"threshold_initialize(): max value must be lesser than maximum image value");

        output_iterator output_begin  = phi.begin();
        output_iterator output_end    = phi.end();
        output_iterator output;
        for(output=output_begin;output!=output_end;++output)
        {
          image_type value = U( output.get_index() );
          if(value >= min_value && value <= max_value)
            *output = -100;
          else
            *output = 100;
        }
      }

    }// end namespace chanvese

    /**
    *
    * @param phi      Input level set.
    * @param U        Input image.
    * @param lambda   Weight of input/output regions (legal values 0..1) 0.5 means input and output are weighted equally.
    * @param mu       Mean Curvature regularization (requires input level set to be close to signed distance grid)
    * @param nu       Area (Well, it is really volume:-) regularization.
    * @param dt       Time-step to use in update.
    * @param psi      Output levelset. Note if input is a signed distance grid then output may not be a signed distance grid, thus you may need to redistance output levelset.
    *
    * @param epsilon          Steady state threshold testing.
    * @param max_iterations   Maximum number of iterations allowed to reach steady state.
    */
    template<
      typename grid_type_in
      , typename image_grid_type
      , typename real_type
      , typename grid_type_out
    >
    inline void chan_vese_auto_in_out(
    grid_type_in  & phi
    , image_grid_type const & U
    , real_type const & lambda
    , real_type const & mu
    , real_type const & nu
    , real_type const & dt
    , grid_type_out & psi
    , real_type const & epsilon = 10e-7
    , size_t const & max_iterations = 10
    )
    {
      typedef typename image_grid_type::value_type       image_type;

      assert(mu>=0     || !"chan_vese_auto_in_out(): curvature coefficient must be non-negative");
      assert(nu>=0     || !"chan_vese_auto_in_out(): area coefficient must be non-negative");
      assert(dt>0      || !"chan_vese_auto_in_out(): time-step must be positive");
      assert(lambda>=0 || !"chan_vese_auto_in_out(): region weight must be non-negative");
      assert(lambda<=1 || !"chan_vese_auto_in_out(): region weight must be less than or equal to one");

      image_type   C_in           = chan_vese::compute_cin(phi,U);
      image_type   C_out          = chan_vese::compute_cout(phi,U);
      size_t unchanged = 0;
      for(size_t iteration = 0;iteration<max_iterations;++iteration)
      {
        if (C_in == C_out)
        {
          // TODO: problems might arise when image_type is unsigned
          //        C_in *= real_type(1.1);
          //        C_out *= real_type(0.9);
          const real_type upper = boost::numeric_cast<real_type>( 1.1 );
          const real_type lower = boost::numeric_cast<real_type>( 0.9 );
          C_in  = boost::numeric_cast<image_type>( C_in* upper   );
          C_out = boost::numeric_cast<image_type>( C_out * lower );
        }
        real_type max_delta = chan_vese::update(phi,U,C_in,C_out,lambda,mu,nu,dt,psi);
        std::cout << "Chan-Vese: Delta max = " << max_delta << std::endl;
        if(max_delta < epsilon )
        {
          std::cout << "Chan-Vese: Steady state reached in " << iteration << " iteration, delta max = " << max_delta << std::endl;
          return;
        }
        phi = psi;
        image_type   C_in_new  = chan_vese::compute_cin(phi,U);
        image_type   C_out_new = chan_vese::compute_cout(phi,U);


        if((C_in == C_in_new)&&(C_out == C_out_new))
        {
          ++unchanged;
          if(unchanged >= 3)
          {
            std::cout << "Chan-Vese: Average region unchanged in 3 iterations" << std::endl;
            return;
          }
        }
        else if(unchanged>0)
          --unchanged;

        C_in = C_in_new;
        C_out = C_out_new;

      }
      std::cout << "Chan-Vese: Maximum iterations reached minimum" << std::endl;
    }

    /**
    *
    * @param phi      Input level set.
    * @param U        Input image.
    * @param C_in     Mean value of inside region. This is the mean value of the region that phi is supposed to enclose.
    * @param C_out    Mean value of outside region. This is the mean value of the region that is supposed to lie outside phi.
    * @param lambda   Weight of input/output regions (legal values 0..1) 0.5 means input and output are weighted equally.
    * @param mu       Mean Curvature regularization (requires input level set to be close to signed distance grid)
    * @param nu       Area (Well, it is really volume:-) regularization.
    * @param dt       Time-step to use in update.
    * @param psi      Output levelset. Note if input is a signed distance grid then output may not be a signed distance grid, thus you may need to redistance output levelset.
    */
    template<
      typename grid_type_in
      , typename image_type
      , typename real_type
      , typename grid_type_out
    >
    inline void chan_vese_fixed_in_out(
    grid_type_in const & phi
    , image_type const & U
    , typename image_type::value_type const & C_in
    , typename image_type::value_type const & C_out
    , real_type const & lambda
    , real_type const & mu
    , real_type const & nu
    , real_type const & dt
    , grid_type_out & psi
    )
    {
      assert(mu>=0     || !"chan_vese_fixed_in_out(): curvature coefficient must be non-negative");
      assert(nu>=0     || !"chan_vese_fixed_in_out(): area coefficient must be non-negative");
      assert(dt>0      || !"chan_vese_fixed_in_out(): time-step must be positive");
      assert(lambda>=0 || !"chan_vese_fixed_in_out(): region weight must be non-negative");
      assert(lambda<=1 || !"chan_vese_fixed_in_out(): region weight must be less than or equal to one");

      chan_vese::update(phi,U,C_in,C_out,lambda,mu,nu,dt,psi);
    }


    /**
    *
    * @param phi      Input level set.
    * @param U        Input image.
    * @param C_in     Mean value of inside region. This is the mean value of the region that phi is supposed to enclose.
    * @param lambda   Weight of input/output regions (legal values 0..1) 0.5 means input and output are weighted equally.
    * @param mu       Mean Curvature regularization (requires input level set to be close to signed distance grid)
    * @param nu       Area (Well, it is really volume:-) regularization.
    * @param dt       Time-step to use in update.
    * @param psi      Output levelset. Note if input is a signed distance grid then output may not be a signed distance grid, thus you may need to redistance output levelset.
    */
    template<
      typename grid_type_in
      , typename image_grid_type
      , typename real_type
      , typename grid_type_out
    >
    inline void chan_vese_fixed_in(
    grid_type_in const & phi
    , image_grid_type const & U
    , typename image_grid_type::value_type C_in
    , real_type const & lambda
    , real_type const & mu
    , real_type const & nu
    , real_type const & dt
    , grid_type_out & psi
    )
    {
      typedef typename image_grid_type::value_type       image_type;

      assert(mu>=0     || !"chan_vese_fixed_in(): curvature coefficient must be non-negative");
      assert(nu>=0     || !"chan_vese_fixed_in(): area coefficient must be non-negative");
      assert(dt>0      || !"chan_vese_fixed_in(): time-step must be positive");
      assert(lambda>=0 || !"chan_vese_fixed_in(): region weight must be non-negative");
      assert(lambda<=1 || !"chan_vese_fixed_in(): region weight must be less than or equal to one");

      image_type  C_out = chan_vese::compute_cout(phi,U);
      if (C_in == C_out)
      {
        // TODO: problems might arise when image_type is unsigned
        //      C_in *= real_type(1.1);
        //      C_out *= real_type(0.9);
        const real_type upper = boost::numeric_cast<real_type>( 1.1 );
        const real_type lower = boost::numeric_cast<real_type>( 0.9 );
        C_in  = boost::numeric_cast<image_type>( C_in* upper   );
        C_out = boost::numeric_cast<image_type>( C_out * lower );
      }
      chan_vese::update(phi,U,C_in,C_out,lambda,mu,nu,dt,psi);
    }


    /**
    *
    * @param phi      Input level set.
    * @param U        Input image.
    * @param C_out    Mean value of outside region. This is the mean value of the region that is supposed to lie outside phi.
    * @param lambda   Weight of input/output regions (legal values 0..1) 0.5 means input and output are weighted equally.
    * @param mu       Mean Curvature regularization (requires input level set to be close to signed distance grid)
    * @param nu       Area (Well, it is really volume:-) regularization.
    * @param dt       Time-step to use in update.
    * @param psi      Output levelset. Note if input is a signed distance grid then output may not be a signed distance grid, thus you may need to redistance output levelset.
    */
    template<
      typename grid_type_in
      , typename image_grid_type
      , typename real_type
      , typename grid_type_out
    >
    inline void chan_vese_fixed_out(
    grid_type_in const & phi
    , image_grid_type const & U
    , typename image_grid_type::value_type C_out
    , real_type const & lambda
    , real_type const & mu
    , real_type const & nu
    , real_type const & dt
    , grid_type_out & psi
    )
    {
      typedef typename image_grid_type::value_type       image_type;

      assert(mu>=0     || !"chan_vese_fixed_out(): curvature coefficient must be non-negative");
      assert(nu>=0     || !"chan_vese_fixed_out(): area coefficient must be non-negative");
      assert(dt>0      || !"chan_vese_fixed_out(): time-step must be positive");
      assert(lambda>=0 || !"chan_vese_fixed_out(): region weight must be non-negative");
      assert(lambda<=1 || !"chan_vese_fixed_out(): region weight must be less than or equal to one");

      image_type  C_in = chan_vese::compute_cin(phi,U);
      if (C_in == C_out)
      {
        // TODO: problems might arise when image_type is unsigned
        //      C_in *= real_type(1.1);
        //      C_out *= real_type(0.9);
        const real_type upper = boost::numeric_cast<real_type>( 1.1 );
        const real_type lower = boost::numeric_cast<real_type>( 0.9 );
        C_in  = boost::numeric_cast<image_type>( C_in* upper   );
        C_out = boost::numeric_cast<image_type>( C_out * lower );
      }
      chan_vese::update(phi,U,C_in,C_out,lambda,mu,nu,dt,psi);
    }

  } // namespace grid
} // namespace OpenTissue

//  OPENTISSUE_CORE_CONTAINERS_GRID_UTIL_CHANVESE_H
#endif
