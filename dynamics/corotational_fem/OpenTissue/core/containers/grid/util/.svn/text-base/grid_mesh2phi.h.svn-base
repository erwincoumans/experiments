#ifndef OPENTISSUE_CORE_CONTAINERS_GRID_UTIL_GRID_MESH2PHI_H
#define OPENTISSUE_CORE_CONTAINERS_GRID_UTIL_GRID_MESH2PHI_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

#include <OpenTissue/core/geometry/t4_gpu_scan/t4_gpu_scan.h>
#include <OpenTissue/core/geometry/t4_cpu_scan/t4_cpu_scan.h>

namespace OpenTissue
{
  namespace grid
  {

    /**
    * Mesh to signed distance field conversion.
    *
    * This function tries to find a reasonable resolution
    * of a signed distance field and uses t4 gpu scan method to
    * compute the signed distance field.
    *
    * @param mesh              A polygonal mesh.
    * @param phi               Upon return this argument contains a signed distance field of the specified mesh.
    * @param max_resolution    This argument can be used to set a maximum upper limit on the resolution of the signed distance field. Default value is 64.
    * @param use_gpu           Boolean flag indicating whether the gpu should be used to compute the signed distance field. Default value is true.
    */
    template<typename mesh_type, typename grid_type>
    inline void mesh2phi(mesh_type & mesh, grid_type & phi, size_t max_resolution = 64, bool use_gpu = true)
    {
      using std::min;
      using std::max;
      using std::sqrt;
      using std::ceil;

      typedef typename mesh_type::math_types                        math_types;
      typedef typename math_types::value_traits                     value_traits;
      typedef typename math_types::vector3_type                     vector3_type;
      typedef typename math_types::real_type                        real_type;

      //--- Compute a suitable grid-spacing for this mesh
      real_type min_area;
      real_type max_area;
      mesh::compute_minmax_face_area(mesh,min_area,max_area);

      assert(min_area>0 || !"mesh2phi(): Minimum face area is zero!");

      //--- first we try to make a best guess at how the grid spacing
      //--- Our strategy is that we ideally want 5x5 grids for the
      //--- minimum quad in the mesh!!!
      real_type delta = sqrt( (2.0*min_area) / 25.0 ); 

      vector3_type min_coord;
      vector3_type max_coord;

      mesh::compute_mesh_minimum_coord(mesh,min_coord);
      mesh::compute_mesh_maximum_coord(mesh,max_coord);

      //--- Compute ``best'' resolution for this mesh
      size_t res_x = boost::numeric_cast<size_t>( ceil(   (max_coord(0)-min_coord(0))/ delta  ) - 1 );
      size_t res_y = boost::numeric_cast<size_t>( ceil(   (max_coord(1)-min_coord(1))/ delta  ) - 1 );
      size_t res_z = boost::numeric_cast<size_t>( ceil(   (max_coord(2)-min_coord(2))/ delta  ) - 1 );
      size_t resolution1 = max(res_x,max(res_y,res_z));
      std::cout << "mesh2phi(): needed resolution = " << resolution1 << std::endl;

      size_t resolution2 = OpenTissue::math::upper_power2(resolution1);// min(m_resolution,resolution);
      std::cout << "mesh2phi(): best power2 resolution = " << resolution2 << std::endl;

      //--- Make sure that resolution is not greather than the one specified by the user...
      size_t resolution3 =  min(max_resolution,resolution2);      
      std::cout << "mesh2phi(): resolution set to = " << resolution3 << std::endl;

      //--- Next we want to make sure that we a layer of grid cells outside the mesh
      //--- First we compute the actual grid spacing (taking into acount that
      //--- we want a boundary of total thickness of 5 cells around the
      //--- enclosing mesh.
      //assert(resolution3 > 7 || !"mesh2phi(): resolution too small");
      resolution3 = max(size_t(8),resolution3);

      real_type  dx = ( max_coord(0)-min_coord(0) )  / (resolution3-6);
      real_type  dy = ( max_coord(1)-min_coord(1) )  / (resolution3-6);
      real_type  dz = ( max_coord(2)-min_coord(2) )  / (resolution3-6);

      vector3_type safety_band(dx,dy,dz);
      min_coord -= 2.0*safety_band;  //--- uneven to break symmetry/alignment between geometry and grid
      max_coord += 3.0*safety_band;

      //--- Finally we create the phi-grid that will end up containing the signed distance grid.
      phi.create(min_coord,max_coord, resolution3, resolution3, resolution3 );

      vector3_type diff = max_coord - min_coord;
      real_type band = sqrt(diff*diff)*value_traits::half();

      mesh::compute_angle_weighted_vertex_normals(mesh);

      if(use_gpu)
      {
        bool gpu_done = t4_gpu_scan(
          mesh
          , band
          , phi
          );
        // Test if we should fall back on CPU 
        if(!gpu_done)
          t4_cpu_scan(mesh,band,phi, t4_cpu_signed() );      
      }
      else
      {
        t4_cpu_scan(mesh,band,phi, t4_cpu_signed() );
      }

      std::cout << "mesh2phi(): completed phi computation" << std::endl;      
    }

    /**
    * Mesh to signed distance field conversion.
    *
    * @param mesh              A polygonal mesh.
    * @param phi               Upon return this argument contains a signed distance field of the specified mesh.
    * @param bandsize          This argument can be used to set the size of a band enclosing the mesh.
    * @param max_resolution    This argument can be used to set the wanted resolution of the resuling distance field.
    * @param use_gpu           Boolean flag indicating whether the gpu should be used to compute the signed distance field. Default value is true.
    */
    template<typename mesh_type, typename grid_type>
    inline void mesh2phi(mesh_type & mesh, grid_type & phi, double bandsize, size_t resolution, bool use_gpu = true)
    {
      using std::sqrt;

      typedef typename mesh_type::math_types                        math_types;
      typedef typename math_types::value_traits                     value_traits;
      typedef typename math_types::vector3_type                     vector3_type;
      typedef typename math_types::real_type                        real_type;

      real_type band = boost::numeric_cast<real_type>(bandsize);
      if(band<=value_traits::zero())
        throw std::invalid_argument("mesh2phi() bandsize must be positive");

      vector3_type min_coord;
      vector3_type max_coord;

      mesh::compute_mesh_minimum_coord(mesh,min_coord);
      mesh::compute_mesh_maximum_coord(mesh,max_coord);

      vector3_type safety_band(band,band,band);
      min_coord -= safety_band;
      max_coord += safety_band;

      //--- Finally we create the phi-grid that will end up containing the signed distance grid.
      phi.create(min_coord,max_coord, resolution, resolution, resolution );

      vector3_type diff = max_coord - min_coord;
      band = sqrt(diff*diff)/value_traits::two();

      mesh::compute_angle_weighted_vertex_normals(mesh);
      if(use_gpu)
      {
        bool gpu_done = t4_gpu_scan(
          mesh
          , band
          , phi
          );
        // Test if we should fall back on CPU 
        if(!gpu_done)
          t4_cpu_scan(mesh,band,phi, t4_cpu_signed() );
      }
      else
      {
        t4_cpu_scan(mesh,band,phi, t4_cpu_signed() );
      }
      std::cout << "mesh2phi(): completed phi computation" << std::endl;      
    }

  } // namespace grid
} // namespace OpenTissue

// OPENTISSUE_CORE_CONTAINERS_GRID_UTIL_GRID_MESH2PHI_H
#endif
