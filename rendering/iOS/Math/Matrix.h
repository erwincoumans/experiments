/*
Oolong Engine for the iPhone / iPod touch
Copyright (c) 2007-2008 Wolfgang Engel  http://code.google.com/p/oolongengine/

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/
// Description: Vector, Matrix and quaternion functions for floating and fixed
//               point math. The general matrix format used is directly compatible
//               with, for example, both DirectX and OpenGL. For the reasons why,
//               read this:
//               http://research.microsoft.com/~hollasch/cgindex/math/matrix/column-vec.html
//

#ifndef MATRIX_H_
#define MATRIX_H_

#include "Vector.h"
#include "Quaternion.h"

#define MAT00 0
#define MAT01 1
#define MAT02 2
#define MAT03 3
#define MAT10 4
#define MAT11 5
#define MAT12 6
#define MAT13 7
#define MAT20 8
#define MAT21 9
#define MAT22 10
#define MAT23 11
#define MAT30 12
#define MAT31 13
#define MAT32 14
#define MAT33 15

/*
struct Matrix4x4 
{ 
    // The elements of the 4x4 matrix are stored in 
    // column-major order (see "OpenGL Programming Guide", 
    // 3rd edition, pp 106, glLoadMatrix). 
    float   _11, _21, _31, _41; 
    float   _12, _22, _32, _42; 
    float   _13, _23, _33, _43; 
    float   _14, _24, _34, _44; 
}; 
*/

#define _11 0
#define _12 1
#define _13 2
#define _14 3
#define _21 4
#define _22 5
#define _23 6
#define _24 7
#define _31 8
#define _32 9
#define _33 10
#define _34 11
#define _41 12
#define _42 13
#define _43 14
#define _44 15

#define _ABS(a)		((a) <= 0 ? -(a) : (a) )

// Useful values
#define PIOVERTWOf	(3.1415926535f / 2.0f)
#define PIf			(3.1415926535f)
#define TWOPIf		(3.1415926535f * 2.0f)
#define ONEf		(1.0f)


//
// 4x4 floating point matrix
//
class MATRIX
{
public:
    float* operator [] ( const int Row )
	{
		return &f[Row<<2];
	}
	float f[16];	/*!< Array of float */
};


//
// Reset matrix to identity
// outputs the identity matrix
//
void MatrixIdentity(MATRIX &mOut);

//
// Multiply mA by mB and assign the result to mOut
// (mOut = p1 * p2). A copy of the result matrix is done in
// the function because mOut can be a parameter mA or mB.
//
void MatrixMultiply(
	MATRIX			&mOut,
	const MATRIX	&mA,
	const MATRIX	&mB);
	
//
// Multiply vector vIn by matrix mIn and assign result to vOut.
// Copies result vector, so vIn and vOut can be the same.
//
void MatrixVec4Multiply(
	VECTOR4			&vOut,
	const VECTOR4	&vIn,
	const MATRIX	&mIn);

//
// Build a translation matrix mOut using fX, fY and fZ.
//
void MatrixTranslation(
	MATRIX	&mOut,
	const float	fX,
	const float	fY,
	const float	fZ);
	

//
// Build a scale matrix mOut using fX, fY and fZ.
//
void MatrixScaling(
	MATRIX	&mOut,
	const float fX,
	const float fY,
	const float fZ);

//
// Create an around-axis rotation matrix mOut. - added by Jimmy
//
void MatrixRotationAxis( MATRIX &mOut, 
						const float fAngle, 	
						const float fX,
						const float fY,
						const float fZ);

//
// Create an X rotation matrix mOut.
//
void MatrixRotationX(
	MATRIX	&mOut,
	const float fAngle);


//
// Create an Y rotation matrix mOut.
//
void MatrixRotationY(
	MATRIX	&mOut,
	const float fAngle);

//
// Create an Z rotation matrix mOut.
//
void MatrixRotationZ(
	MATRIX	&mOut,
	const float fAngle);


//
// Compute the transpose matrix of mIn.
//
void MatrixTranspose(
	MATRIX			&mOut,
	const MATRIX	&mIn);


//
// Compute the inverse matrix of mIn.
//	The matrix must be of the form :
//	A 0
//	C 1
// Where A is a 3x3 matrix and C is a 1x3 matrix.
//
void MatrixInverse(
	MATRIX			&mOut,
	const MATRIX	&mIn);


//
// Compute the inverse matrix of mIn.
// Uses a linear equation solver and the knowledge that M.M^-1=I.
// Use this fn to calculate the inverse of matrices that
// MatrixInverse() cannot.
//
void MatrixInverseEx(
	MATRIX			&mOut,
	const MATRIX	&mIn);


//
// Create a look-at view matrix.
//
void MatrixLookAtLH(
	MATRIX			&mOut,
	const VECTOR3	&vEye,
	const VECTOR3	&vAt,
	const VECTOR3	&vUp);

// 
// Create a look-at view matrix.
//
void MatrixLookAtRH(
	MATRIX			&mOut,
	const VECTOR3	&vEye,
	const VECTOR3	&vAt,
	const VECTOR3	&vUp);


void MatrixPerspectiveFovLH(
	MATRIX	&mOut,
	const float	fFOVy,
	const float	fAspect,
	const float	fNear,
	const float	fFar,
	const bool  bRotate = false);


void MatrixPerspectiveFovRH(
	MATRIX	&mOut,
	const float	fFOVy,
	const float	fAspect,
	const float	fNear,
	const float	fFar,
	const bool  bRotate = false);


void MatrixOrthoLH(
	MATRIX	&mOut,
	const float w,
	const float h,
	const float zn,
	const float zf,
	const bool  bRotate = false);


void MatrixOrthoRH(
	MATRIX	&mOut,
	const float w,
	const float h,
	const float zn,
	const float zf,
	const bool  bRotate = false);

void MatrixVec3Multiply(VECTOR3		&vOut,
						const VECTOR3	&vIn,
						const MATRIX	&mIn);

void MatrixVec3Lerp(
	VECTOR3		&vOut,
	const VECTOR3	&v1,
	const VECTOR3	&v2,
	const float			s);


float MatrixVec3DotProduct(
	const VECTOR3	&v1,
	const VECTOR3	&v2);


void MatrixVec3CrossProduct(
	VECTOR3		&vOut,
	const VECTOR3	&v1,
	const VECTOR3	&v2);


void MatrixVec3Normalize(
	VECTOR3		&vOut,
	const VECTOR3	&vIn);

void MatrixVec4Normalize(
	VECTOR4		&vOut,
	const VECTOR4	&vIn);

float MatrixVec3Length(
	const VECTOR3	&vIn);


void MatrixQuaternionIdentity(
	QUATERNION		&qOut);


void MatrixQuaternionRotationAxis(
	QUATERNION		&qOut,
	const VECTOR3	&vAxis,
	const float			fAngle);

void MatrixQuaternionToAxisAngle(
	const QUATERNION	&qIn,
	VECTOR3			&vAxis,
	float					&fAngle);


void MatrixQuaternionSlerp(
	QUATERNION			&qOut,
	const QUATERNION	&qA,
	const QUATERNION	&qB,
	const float				t);


void MatrixQuaternionNormalize(QUATERNION &quat);

//
// Create rotation matrix from submitted quaternion.
// Assuming the quaternion is of the form [X Y Z W]:
//
//						|       2     2									|
//						| 1 - 2Y  - 2Z    2XY - 2ZW      2XZ + 2YW		 0	|
//						|													|
//						|                       2     2					|
//					M = | 2XY + 2ZW       1 - 2X  - 2Z   2YZ - 2XW		 0	|
//						|													|
//						|                                      2     2		|
//						| 2XZ - 2YW       2YZ + 2XW      1 - 2X  - 2Y	 0	|
//						|													|
//						|     0			   0			  0          1  |
//
void MatrixRotationQuaternion(
	MATRIX				&mOut,
	const QUATERNION	&quat);

void MatrixQuaternionMultiply(
	QUATERNION			&qOut,
	const QUATERNION	&qA,
	const QUATERNION	&qB);

//
// Solves 'nCnt' simultaneous equations of 'nCnt' variables.
// pRes should be an array large enough to contain the
// results: the values of the 'nCnt' variables.
// This fn recursively uses Gaussian Elimination.
//
void MatrixLinearEqSolve(
	float		* const pRes,
	float		** const pSrc,
	const int	nCnt);

#endif // MATRIX_H_

