#ifndef OPENTISSUE_UTILITY_GL_MATRIX_UTIL_H
#define OPENTISSUE_UTILITY_GL_MATRIX_UTIL_H
//
// OpenTissue Template Library
// - A generic toolbox for physics-based modeling and simulation.
// Copyright (C) 2008 Department of Computer Science, University of Copenhagen.
//
// OTTL is licensed under zlib: http://opensource.org/licenses/zlib-license.php
//
#include <OpenTissue/configuration.h>

#include <OpenTissue/utility/gl/gl.h>
#include <cassert>

namespace OpenTissue
{

  namespace gl
  {

    /**
    * OpenGL Matrix-Vector Transformation.
    *
    * @param M         OpenGL matrix type in column major format.
    * @param v         A 3 dimensional vector type.
    *
    * @return          The homegenoized result of M*[v^T,1]^T
    */
    template<typename real_type, typename vector3_type>
    inline vector3_type xform( real_type * M, vector3_type const &  v )
    {
      real_type x = (M[ 0 ] * v[ 0 ]) + (M[ 4 ] * v[ 1 ]) + (M[ 8  ] * v[ 2 ]) + M[ 12 ];
      real_type y = (M[ 1 ] * v[ 0 ]) + (M[ 5 ] * v[ 1 ]) + (M[ 9  ] * v[ 2 ]) + M[ 13 ];
      real_type z = (M[ 2 ] * v[ 0 ]) + (M[ 6 ] * v[ 1 ]) + (M[ 10 ] * v[ 2 ]) + M[ 14 ];
      real_type w = (M[ 3 ] * v[ 0 ]) + (M[ 7 ] * v[ 1 ]) + (M[ 11 ] * v[ 2 ]) + M[ 15 ];
      assert(w || !"mul(M,v): w was zero");
      return vector3_type( x / w, y / w, z / w);
    }

    /**
    * Matrix-Matrix Multiplication.
    * Auxiliary method used to multiply opengl matrices.
    *
    *    C = A*B
    *
    * All matrices are represented in column major form.
    *
    * @param A
    * @param B
    * @param C
    */
    template<typename real_type>
    inline void mul(real_type const* A,real_type const * B,real_type * C)
    {
      assert(A!=C || !"mul(A,B,C): A and C matrices must be different");
      assert(B!=C || !"mul(A,B,C): B and C matrices must be different");

      C[ 0] =  A[0]*B[ 0] + A[4]*B[ 1] + A[ 8]*B[ 2] + A[12]*B[ 3];
      C[ 1] =  A[1]*B[ 0] + A[5]*B[ 1] + A[ 9]*B[ 2] + A[13]*B[ 3];
      C[ 2] =  A[2]*B[ 0] + A[6]*B[ 1] + A[10]*B[ 2] + A[14]*B[ 3];
      C[ 3] =  A[3]*B[ 0] + A[7]*B[ 1] + A[11]*B[ 2] + A[15]*B[ 3];
      C[ 4] =  A[0]*B[ 4] + A[4]*B[ 5] + A[ 8]*B[ 6] + A[12]*B[ 7];
      C[ 5] =  A[1]*B[ 4] + A[5]*B[ 5] + A[ 9]*B[ 6] + A[13]*B[ 7];
      C[ 6] =  A[2]*B[ 4] + A[6]*B[ 5] + A[10]*B[ 6] + A[14]*B[ 7];
      C[ 7] =  A[3]*B[ 4] + A[7]*B[ 5] + A[11]*B[ 6] + A[15]*B[ 7];
      C[ 8] =  A[0]*B[ 8] + A[4]*B[ 9] + A[ 8]*B[10] + A[12]*B[11];
      C[ 9] =  A[1]*B[ 8] + A[5]*B[ 9] + A[ 9]*B[10] + A[13]*B[11];
      C[10] =  A[2]*B[ 8] + A[6]*B[ 9] + A[10]*B[10] + A[14]*B[11];
      C[11] =  A[3]*B[ 8] + A[7]*B[ 9] + A[11]*B[10] + A[15]*B[11];
      C[12] =  A[0]*B[12] + A[4]*B[13] + A[ 8]*B[14] + A[12]*B[15];
      C[13] =  A[1]*B[12] + A[5]*B[13] + A[ 9]*B[14] + A[13]*B[15];
      C[14] =  A[2]*B[12] + A[6]*B[13] + A[10]*B[14] + A[14]*B[15];
      C[15] =  A[3]*B[12] + A[7]*B[13] + A[11]*B[14] + A[15]*B[15];
    }

    /**
    * Transpose a 4x4 matrix
    *
    * @param M   the matrix to be transposed
    * @param T   The resulting transposed matrix.
    */
    template<typename real_type>
    inline void transpose( real_type const * M, real_type * T )
    {
      assert(M!=T  || !"transpose(): M and T must be two different matrices!" );

      T[0] = M[0];    T[4] = M[1];    T[8]  = M[2];     T[12] = M[3];
      T[1] = M[4];    T[5] = M[5];    T[9]  = M[6];     T[13] = M[7];
      T[2] = M[8];    T[6] = M[9];    T[10] = M[10];    T[14] = M[11];
      T[3] = M[12];   T[7] = M[13];   T[11] = M[14];    T[15] = M[15];

    }

    /**
    * Transpose a 4x4 matrix
    *
    * @param M   The matrix to be transposed
    */
    template<typename real_type>
    inline void transpose( real_type * M )
    {
      real_type temp;
      /* self transpose */
      temp = M[ 0 * 4 + 1 ];
      M[ 0 * 4 + 1 ] = M[ 1 * 4 + 0 ];
      M[ 1 * 4 + 0 ] = temp;
      temp = M[ 0 * 4 + 2 ];
      M[ 0 * 4 + 2 ] = M[ 2 * 4 + 0 ];
      M[ 2 * 4 + 0 ] = temp;
      temp = M[ 0 * 4 + 3 ];
      M[ 0 * 4 + 3 ] = M[ 3 * 4 + 0 ];
      M[ 3 * 4 + 0 ] = temp;

      temp = M[ 1 * 4 + 2 ];
      M[ 1 * 4 + 2 ] = M[ 2 * 4 + 1 ];
      M[ 2 * 4 + 1 ] = temp;

      temp = M[ 1 * 4 + 3 ];
      M[ 1 * 4 + 3 ] = M[ 3 * 4 + 1 ];
      M[ 3 * 4 + 1 ] = temp;

      temp = M[ 2 * 4 + 3 ];
      M[ 2 * 4 + 3 ] = M[ 3 * 4 + 2 ];
      M[ 3 * 4 + 2 ] = temp;
    }

    /**
    * Invert a 4x4 matrix
    *
    * @param M   the matrix to be inverted
    * @param iM  the inverted matrix
    *
    * @return    the determinant of M. If zero, the matrix is invertible.
    */
    template<typename real_type>
    inline real_type invert( real_type const * M, real_type * I )
    {
      assert(M!=I  || !"invert(): M and I must be two different matrices!" );

      real_type const & a = M[ 0 ];    real_type const & e = M[ 1 ];    real_type const & i = M[ 2  ];    real_type const & m = M[ 3 ];
      real_type const & b = M[ 4 ];    real_type const & f = M[ 5 ];    real_type const & j = M[ 6  ];    real_type const & n = M[ 7 ];
      real_type const & c = M[ 8 ];    real_type const & g = M[ 9 ];    real_type const & k = M[ 10 ];    real_type const & o = M[ 11 ];
      real_type const & d = M[ 12 ];   real_type const & h = M[ 13 ];   real_type const & l = M[ 14 ];    real_type const & p = M[ 15 ];

      real_type det = d * g * j * m - c * h * j * m - d * f * k * m + b * h * k * m + c * f * l * m -
        b * g * l * m - d * g * i * n + c * h * i * n + d * e * k * n - a * h * k * n -
        c * e * l * n + a * g * l * n + d * f * i * o - b * h * i * o - d * e * j * o +
        a * h * j * o + b * e * l * o - a * f * l * o - c * f * i * p + b * g * i * p +
        c * e * j * p - a * g * j * p - b * e * k * p + a * f * k * p;

      if ( !det )
      {
        std::cout <<  "invert(): Cannot invert matrix (determinant is zero)." << std::endl;
        return real_type(0.0);
      }

      real_type inv_det = 1.0 / det;

      I[ 0 * 4 + 0 ] = ( -( h * k * n ) + g * l * n + h * j * o - f * l * o - g * j * p + f * k * p ) * inv_det;
      I[ 1 * 4 + 0 ] = ( d * k * n - c * l * n - d * j * o + b * l * o + c * j * p - b * k * p ) * inv_det;
      I[ 2 * 4 + 0 ] = ( -( d * g * n ) + c * h * n + d * f * o - b * h * o - c * f * p + b * g * p ) * inv_det;
      I[ 3 * 4 + 0 ] = ( d * g * j - c * h * j - d * f * k + b * h * k + c * f * l - b * g * l ) * inv_det;
      I[ 0 * 4 + 1 ] = ( h * k * m - g * l * m - h * i * o + e * l * o + g * i * p - e * k * p ) * inv_det;
      I[ 1 * 4 + 1 ] = ( -( d * k * m ) + c * l * m + d * i * o - a * l * o - c * i * p + a * k * p ) * inv_det;
      I[ 2 * 4 + 1 ] = ( d * g * m - c * h * m - d * e * o + a * h * o + c * e * p - a * g * p ) * inv_det;
      I[ 3 * 4 + 1 ] = ( -( d * g * i ) + c * h * i + d * e * k - a * h * k - c * e * l + a * g * l ) * inv_det;
      I[ 0 * 4 + 2 ] = ( -( h * j * m ) + f * l * m + h * i * n - e * l * n - f * i * p + e * j * p ) * inv_det;
      I[ 1 * 4 + 2 ] = ( d * j * m - b * l * m - d * i * n + a * l * n + b * i * p - a * j * p ) * inv_det;
      I[ 2 * 4 + 2 ] = ( -( d * f * m ) + b * h * m + d * e * n - a * h * n - b * e * p + a * f * p ) * inv_det;
      I[ 3 * 4 + 2 ] = ( d * f * i - b * h * i - d * e * j + a * h * j + b * e * l - a * f * l ) * inv_det;
      I[ 0 * 4 + 3 ] = ( g * j * m - f * k * m - g * i * n + e * k * n + f * i * o - e * j * o ) * inv_det;
      I[ 1 * 4 + 3 ] = ( -( c * j * m ) + b * k * m + c * i * n - a * k * n - b * i * o + a * j * o ) * inv_det;
      I[ 2 * 4 + 3 ] = ( c * f * m - b * g * m - c * e * n + a * g * n + b * e * o - a * f * o ) * inv_det;
      I[ 3 * 4 + 3 ] = ( -( c * f * i ) + b * g * i + c * e * j - a * g * j - b * e * k + a * f * k ) * inv_det;

      return det;
    }


    /**
    * Orthonormalize OpenGL Matrix
    *
    * @param M     The input matrix in column major format.
    * @param O     The input matrix in column major format.
    */
    template<typename real_type>
    inline void orthonormalize( real_type const * M, real_type * O )
    {
      using std::sqrt;

      real_type len, temp[ 3 ][ 3 ];

      temp[ 0 ][ 0 ] = M[ 0 * 4 + 0 ];
      temp[ 0 ][ 1 ] = M[ 0 * 4 + 1 ];
      temp[ 0 ][ 2 ] = M[ 0 * 4 + 2 ];
      temp[ 1 ][ 0 ] = M[ 1 * 4 + 0 ];
      temp[ 1 ][ 1 ] = M[ 1 * 4 + 1 ];
      temp[ 1 ][ 2 ] = M[ 1 * 4 + 2 ];

      /* normalize x */
      len = sqrt ( temp[ 0 ][ 0 ] * temp[ 0 ][ 0 ] + temp[ 0 ][ 1 ] * temp[ 0 ][ 1 ] + temp[ 0 ][ 2 ] * temp[ 0 ][ 2 ] );
      len = ( len == 0.0f ) ? 1.0f : 1.0f / len;
      temp[ 0 ][ 0 ] *= len;
      temp[ 0 ][ 1 ] *= len;
      temp[ 0 ][ 2 ] *= len;

      /* z = x cross y */
      temp[ 2 ][ 0 ] = temp[ 0 ][ 1 ] * temp[ 1 ][ 2 ] - temp[ 0 ][ 2 ] * temp[ 1 ][ 1 ];
      temp[ 2 ][ 1 ] = temp[ 0 ][ 2 ] * temp[ 1 ][ 0 ] - temp[ 0 ][ 0 ] * temp[ 1 ][ 2 ];
      temp[ 2 ][ 2 ] = temp[ 0 ][ 0 ] * temp[ 1 ][ 1 ] - temp[ 0 ][ 1 ] * temp[ 1 ][ 0 ];

      /* normalize z */
      len = sqrt ( temp[ 2 ][ 0 ] * temp[ 2 ][ 0 ] + temp[ 2 ][ 1 ] * temp[ 2 ][ 1 ] + temp[ 2 ][ 2 ] * temp[ 2 ][ 2 ] );
      len = ( len == 0.0f ) ? 1.0f : 1.0f / len;
      temp[ 2 ][ 0 ] *= len;
      temp[ 2 ][ 1 ] *= len;
      temp[ 2 ][ 2 ] *= len;

      /* y = z cross x */
      temp[ 1 ][ 0 ] = temp[ 2 ][ 1 ] * temp[ 0 ][ 2 ] - temp[ 2 ][ 2 ] * temp[ 0 ][ 1 ];
      temp[ 1 ][ 1 ] = temp[ 2 ][ 2 ] * temp[ 0 ][ 0 ] - temp[ 2 ][ 0 ] * temp[ 0 ][ 2 ];
      temp[ 1 ][ 2 ] = temp[ 2 ][ 0 ] * temp[ 0 ][ 1 ] - temp[ 2 ][ 1 ] * temp[ 0 ][ 0 ];

      /* normalize y */
      len = sqrt ( temp[ 1 ][ 0 ] * temp[ 1 ][ 0 ] + temp[ 1 ][ 1 ] * temp[ 1 ][ 1 ] + temp[ 1 ][ 2 ] * temp[ 1 ][ 2 ] );
      len = ( len == 0.0f ) ? 1.0f : 1.0f / len;
      temp[ 1 ][ 0 ] *= len;
      temp[ 1 ][ 1 ] *= len;
      temp[ 1 ][ 2 ] *= len;

      /* update matrix4 */
      O[ 0 * 4 + 0 ] = temp[ 0 ][ 0 ];
      O[ 0 * 4 + 1 ] = temp[ 0 ][ 1 ];
      O[ 0 * 4 + 2 ] = temp[ 0 ][ 2 ];
      O[ 1 * 4 + 0 ] = temp[ 1 ][ 0 ];
      O[ 1 * 4 + 1 ] = temp[ 1 ][ 1 ];
      O[ 1 * 4 + 2 ] = temp[ 1 ][ 2 ];
      O[ 2 * 4 + 0 ] = temp[ 2 ][ 0 ];
      O[ 2 * 4 + 1 ] = temp[ 2 ][ 1 ];
      O[ 2 * 4 + 2 ] = temp[ 2 ][ 2 ];
    }

  } // namespace gl

} // namespace OpenTissue

//OPENTISSUE_UTILITY_GL_MATRIX_UTIL_H
#endif
