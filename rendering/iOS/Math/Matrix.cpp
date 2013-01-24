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
//
// Description: Vector, Matrix and quaternion functions for floating and fixed
//               point math. The general matrix format used is directly compatible
//               with, for example, both DirectX and OpenGL. For the reasons why,
//               read this:
//               http://research.microsoft.com/~hollasch/cgindex/math/matrix/column-vec.html
//

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

//#include "FixedPoint.h"		// Only needed for trig function float lookups
#include "Matrix.h"
#include "Macros.h"

#ifdef __APPLE__
#include <TargetConditionals.h>
#if (TARGET_IPHONE_SIMULATOR == 0) && (TARGET_OS_IPHONE == 1)
#include "vfpmath/matrix_impl.h"
#endif
#endif

static const MATRIX	c_mIdentity = {
	{
	1, 0, 0, 0,
	0, 1, 0, 0,
	0, 0, 1, 0,
	0, 0, 0, 1
	}
};

void MatrixIdentity(MATRIX &mOut)
{
	mOut.f[ 0]=1.0f;	mOut.f[ 4]=0.0f;	mOut.f[ 8]=0.0f;	mOut.f[12]=0.0f;
	mOut.f[ 1]=0.0f;	mOut.f[ 5]=1.0f;	mOut.f[ 9]=0.0f;	mOut.f[13]=0.0f;
	mOut.f[ 2]=0.0f;	mOut.f[ 6]=0.0f;	mOut.f[10]=1.0f;	mOut.f[14]=0.0f;
	mOut.f[ 3]=0.0f;	mOut.f[ 7]=0.0f;	mOut.f[11]=0.0f;	mOut.f[15]=1.0f;
}


void MatrixMultiply(
	MATRIX			&mOut,
	const MATRIX	&mA,
	const MATRIX	&mB)
{
	MATRIX mRet;

	// Perform calculation on a dummy matrix (mRet)
	mRet.f[ 0] = mA.f[ 0]*mB.f[ 0] + mA.f[ 1]*mB.f[ 4] + mA.f[ 2]*mB.f[ 8] + mA.f[ 3]*mB.f[12];
	mRet.f[ 1] = mA.f[ 0]*mB.f[ 1] + mA.f[ 1]*mB.f[ 5] + mA.f[ 2]*mB.f[ 9] + mA.f[ 3]*mB.f[13];
	mRet.f[ 2] = mA.f[ 0]*mB.f[ 2] + mA.f[ 1]*mB.f[ 6] + mA.f[ 2]*mB.f[10] + mA.f[ 3]*mB.f[14];
	mRet.f[ 3] = mA.f[ 0]*mB.f[ 3] + mA.f[ 1]*mB.f[ 7] + mA.f[ 2]*mB.f[11] + mA.f[ 3]*mB.f[15];

	mRet.f[ 4] = mA.f[ 4]*mB.f[ 0] + mA.f[ 5]*mB.f[ 4] + mA.f[ 6]*mB.f[ 8] + mA.f[ 7]*mB.f[12];
	mRet.f[ 5] = mA.f[ 4]*mB.f[ 1] + mA.f[ 5]*mB.f[ 5] + mA.f[ 6]*mB.f[ 9] + mA.f[ 7]*mB.f[13];
	mRet.f[ 6] = mA.f[ 4]*mB.f[ 2] + mA.f[ 5]*mB.f[ 6] + mA.f[ 6]*mB.f[10] + mA.f[ 7]*mB.f[14];
	mRet.f[ 7] = mA.f[ 4]*mB.f[ 3] + mA.f[ 5]*mB.f[ 7] + mA.f[ 6]*mB.f[11] + mA.f[ 7]*mB.f[15];

	mRet.f[ 8] = mA.f[ 8]*mB.f[ 0] + mA.f[ 9]*mB.f[ 4] + mA.f[10]*mB.f[ 8] + mA.f[11]*mB.f[12];
	mRet.f[ 9] = mA.f[ 8]*mB.f[ 1] + mA.f[ 9]*mB.f[ 5] + mA.f[10]*mB.f[ 9] + mA.f[11]*mB.f[13];
	mRet.f[10] = mA.f[ 8]*mB.f[ 2] + mA.f[ 9]*mB.f[ 6] + mA.f[10]*mB.f[10] + mA.f[11]*mB.f[14];
	mRet.f[11] = mA.f[ 8]*mB.f[ 3] + mA.f[ 9]*mB.f[ 7] + mA.f[10]*mB.f[11] + mA.f[11]*mB.f[15];

	mRet.f[12] = mA.f[12]*mB.f[ 0] + mA.f[13]*mB.f[ 4] + mA.f[14]*mB.f[ 8] + mA.f[15]*mB.f[12];
	mRet.f[13] = mA.f[12]*mB.f[ 1] + mA.f[13]*mB.f[ 5] + mA.f[14]*mB.f[ 9] + mA.f[15]*mB.f[13];
	mRet.f[14] = mA.f[12]*mB.f[ 2] + mA.f[13]*mB.f[ 6] + mA.f[14]*mB.f[10] + mA.f[15]*mB.f[14];
	mRet.f[15] = mA.f[12]*mB.f[ 3] + mA.f[13]*mB.f[ 7] + mA.f[14]*mB.f[11] + mA.f[15]*mB.f[15];

	// Copy result in pResultMatrix
	mOut = mRet;
 }

void MatrixVec4Multiply(VECTOR4			&vOut,
						const VECTOR4	&vIn,
						const MATRIX	&mIn)
{
#if (TARGET_IPHONE_SIMULATOR == 0) && (TARGET_OS_IPHONE == 1)
		Matrix4Vector4Mul(mIn.f,
						  &vIn.x,
						  &vOut.x);
#else
	VECTOR4 result;
	
	/* Perform calculation on a dummy VECTOR (result) */
	result.x = mIn.f[_11] * vIn.x + mIn.f[_21] * vIn.y + mIn.f[_31] * vIn.z + mIn.f[_41] * vIn.w;
	result.y = mIn.f[_12] * vIn.x + mIn.f[_22] * vIn.y + mIn.f[_32] * vIn.z + mIn.f[_42] * vIn.w;
	result.z = mIn.f[_13] * vIn.x + mIn.f[_23] * vIn.y + mIn.f[_33] * vIn.z + mIn.f[_43] * vIn.w;
	result.w = mIn.f[_14] * vIn.x + mIn.f[_24] * vIn.y + mIn.f[_34] * vIn.z + mIn.f[_44] * vIn.w;
	
	vOut = result;
#endif
}


void MatrixTranslation(
	MATRIX	&mOut,
	const float fX,
	const float fY,
	const float fZ)
{
	mOut.f[ 0]=1.0f;	mOut.f[ 4]=0.0f;	mOut.f[ 8]=0.0f;	mOut.f[12]=fX;
	mOut.f[ 1]=0.0f;	mOut.f[ 5]=1.0f;	mOut.f[ 9]=0.0f;	mOut.f[13]=fY;
	mOut.f[ 2]=0.0f;	mOut.f[ 6]=0.0f;	mOut.f[10]=1.0f;	mOut.f[14]=fZ;
	mOut.f[ 3]=0.0f;	mOut.f[ 7]=0.0f;	mOut.f[11]=0.0f;	mOut.f[15]=1.0f;
}


void MatrixScaling(
	MATRIX	&mOut,
	const float fX,
	const float fY,
	const float fZ)
{
	mOut.f[ 0]=fX;		mOut.f[ 4]=0.0f;	mOut.f[ 8]=0.0f;	mOut.f[12]=0.0f;
	mOut.f[ 1]=0.0f;	mOut.f[ 5]=fY;		mOut.f[ 9]=0.0f;	mOut.f[13]=0.0f;
	mOut.f[ 2]=0.0f;	mOut.f[ 6]=0.0f;	mOut.f[10]=fZ;		mOut.f[14]=0.0f;
	mOut.f[ 3]=0.0f;	mOut.f[ 7]=0.0f;	mOut.f[11]=0.0f;	mOut.f[15]=1.0f;
}

void MatrixRotationAxis( MATRIX &mOut, 
						const float fAngle, 	
						const float fX,
						const float fY,
						const float fZ) 
{
	Vec3 axis(fX, fY, fZ);
	axis.normalize();
	float s = (float)sin(fAngle);
	float c = (float)cos(fAngle);
	float x, y, z;
	
	x = axis.x;
	y = axis.y;
	z = axis.z;
	
	mOut.f[ 0] = x * x * (1 - c) + c;
	mOut.f[ 4] = x * y * (1 - c) - (z * s);
	mOut.f[ 8] = x * z * (1 - c) + (y * s);
	mOut.f[12] = 0;
	
	mOut.f[ 1] = y * x * (1 - c) + (z * s);
	mOut.f[ 5] = y * y * (1 - c) + c;
	mOut.f[ 9] = y * z * (1 - c) - (x * s);
	mOut.f[13] = 0;
	
	mOut.f[ 2] = z * x * (1 - c) - (y * s);
	mOut.f[ 6] = z * y * (1 - c) + (x * s);
	mOut.f[10] = z * z * (1 - c) + c;
	mOut.f[14] = 0.0f;
	
	mOut.f[ 3] = 0.0f;
	mOut.f[ 7] = 0.0f;
	mOut.f[11] = 0.0f;
	mOut.f[15] = 1.0f;
}




void MatrixRotationX(
	MATRIX	&mOut,
	const float fAngle)
{
	float		fCosine, fSine;

    /* Precompute cos and sin */
#if defined(BUILD_DX9) || defined(BUILD_D3DM)
	fCosine	= (float)cos(-fAngle);
    fSine	= (float)sin(-fAngle);
#else
	fCosine	= (float)cos(fAngle);
    fSine	= (float)sin(fAngle);
#endif

	/* Create the trigonometric matrix corresponding to X Rotation */
	mOut.f[ 0]=1.0f;	mOut.f[ 4]=0.0f;	mOut.f[ 8]=0.0f;	mOut.f[12]=0.0f;
	mOut.f[ 1]=0.0f;	mOut.f[ 5]=fCosine;	mOut.f[ 9]=fSine;	mOut.f[13]=0.0f;
	mOut.f[ 2]=0.0f;	mOut.f[ 6]=-fSine;	mOut.f[10]=fCosine;	mOut.f[14]=0.0f;
	mOut.f[ 3]=0.0f;	mOut.f[ 7]=0.0f;	mOut.f[11]=0.0f;	mOut.f[15]=1.0f;
}


void MatrixRotationY(
	MATRIX	&mOut,
	const float fAngle)
{
	float		fCosine, fSine;

	/* Precompute cos and sin */
#if defined(BUILD_DX9) || defined(BUILD_D3DM)
	fCosine	= (float)cos(-fAngle);
    fSine	= (float)sin(-fAngle);
#else
	fCosine	= (float)cos(fAngle);
    fSine	= (float)sin(fAngle);
#endif

	/* Create the trigonometric matrix corresponding to Y Rotation */
	mOut.f[ 0]=fCosine;		mOut.f[ 4]=0.0f;	mOut.f[ 8]=-fSine;		mOut.f[12]=0.0f;
	mOut.f[ 1]=0.0f;		mOut.f[ 5]=1.0f;	mOut.f[ 9]=0.0f;		mOut.f[13]=0.0f;
	mOut.f[ 2]=fSine;		mOut.f[ 6]=0.0f;	mOut.f[10]=fCosine;		mOut.f[14]=0.0f;
	mOut.f[ 3]=0.0f;		mOut.f[ 7]=0.0f;	mOut.f[11]=0.0f;		mOut.f[15]=1.0f;
}


void MatrixRotationZ(
	MATRIX	&mOut,
	const float fAngle)
{
	float		fCosine, fSine;

	/* Precompute cos and sin */
#if defined(BUILD_DX9) || defined(BUILD_D3DM)
	fCosine =	(float)cos(-fAngle);
    fSine =		(float)sin(-fAngle);
#else
	fCosine =	(float)cos(fAngle);
    fSine =		(float)sin(fAngle);
#endif

	/* Create the trigonometric matrix corresponding to Z Rotation */
	mOut.f[ 0]=fCosine;		mOut.f[ 4]=fSine;	mOut.f[ 8]=0.0f;	mOut.f[12]=0.0f;
	mOut.f[ 1]=-fSine;		mOut.f[ 5]=fCosine;	mOut.f[ 9]=0.0f;	mOut.f[13]=0.0f;
	mOut.f[ 2]=0.0f;		mOut.f[ 6]=0.0f;	mOut.f[10]=1.0f;	mOut.f[14]=0.0f;
	mOut.f[ 3]=0.0f;		mOut.f[ 7]=0.0f;	mOut.f[11]=0.0f;	mOut.f[15]=1.0f;
}


void MatrixTranspose(
	MATRIX			&mOut,
	const MATRIX	&mIn)
{
	MATRIX	mTmp;

	mTmp.f[ 0]=mIn.f[ 0];	mTmp.f[ 4]=mIn.f[ 1];	mTmp.f[ 8]=mIn.f[ 2];	mTmp.f[12]=mIn.f[ 3];
	mTmp.f[ 1]=mIn.f[ 4];	mTmp.f[ 5]=mIn.f[ 5];	mTmp.f[ 9]=mIn.f[ 6];	mTmp.f[13]=mIn.f[ 7];
	mTmp.f[ 2]=mIn.f[ 8];	mTmp.f[ 6]=mIn.f[ 9];	mTmp.f[10]=mIn.f[10];	mTmp.f[14]=mIn.f[11];
	mTmp.f[ 3]=mIn.f[12];	mTmp.f[ 7]=mIn.f[13];	mTmp.f[11]=mIn.f[14];	mTmp.f[15]=mIn.f[15];

	mOut = mTmp;
}



void MatrixInverse(
	MATRIX			&mOut,
	const MATRIX	&mIn)
{
	MATRIX	mDummyMatrix;
	double		det_1;
	double		pos, neg, temp;

    /* Calculate the determinant of submatrix A and determine if the
       the matrix is singular as limited by the double precision
       floating-point data representation. */
    pos = neg = 0.0;
    temp =  mIn.f[ 0] * mIn.f[ 5] * mIn.f[10];
    if (temp >= 0.0) pos += temp; else neg += temp;
    temp =  mIn.f[ 4] * mIn.f[ 9] * mIn.f[ 2];
    if (temp >= 0.0) pos += temp; else neg += temp;
    temp =  mIn.f[ 8] * mIn.f[ 1] * mIn.f[ 6];
    if (temp >= 0.0) pos += temp; else neg += temp;
    temp = -mIn.f[ 8] * mIn.f[ 5] * mIn.f[ 2];
    if (temp >= 0.0) pos += temp; else neg += temp;
    temp = -mIn.f[ 4] * mIn.f[ 1] * mIn.f[10];
    if (temp >= 0.0) pos += temp; else neg += temp;
    temp = -mIn.f[ 0] * mIn.f[ 9] * mIn.f[ 6];
    if (temp >= 0.0) pos += temp; else neg += temp;
    det_1 = pos + neg;

    /* Is the submatrix A singular? */
    if ((det_1 == 0.0) || (_ABS(det_1 / (pos - neg)) < 1.0e-15))
	{
        /* Matrix M has no inverse */
        printf("Matrix has no inverse : singular matrix\n");
        return;
    }
    else
	{
        /* Calculate inverse(A) = adj(A) / det(A) */
        det_1 = 1.0 / det_1;
        mDummyMatrix.f[ 0] =   ( mIn.f[ 5] * mIn.f[10] - mIn.f[ 9] * mIn.f[ 6] ) * (float)det_1;
        mDummyMatrix.f[ 1] = - ( mIn.f[ 1] * mIn.f[10] - mIn.f[ 9] * mIn.f[ 2] ) * (float)det_1;
        mDummyMatrix.f[ 2] =   ( mIn.f[ 1] * mIn.f[ 6] - mIn.f[ 5] * mIn.f[ 2] ) * (float)det_1;
        mDummyMatrix.f[ 4] = - ( mIn.f[ 4] * mIn.f[10] - mIn.f[ 8] * mIn.f[ 6] ) * (float)det_1;
        mDummyMatrix.f[ 5] =   ( mIn.f[ 0] * mIn.f[10] - mIn.f[ 8] * mIn.f[ 2] ) * (float)det_1;
        mDummyMatrix.f[ 6] = - ( mIn.f[ 0] * mIn.f[ 6] - mIn.f[ 4] * mIn.f[ 2] ) * (float)det_1;
        mDummyMatrix.f[ 8] =   ( mIn.f[ 4] * mIn.f[ 9] - mIn.f[ 8] * mIn.f[ 5] ) * (float)det_1;
        mDummyMatrix.f[ 9] = - ( mIn.f[ 0] * mIn.f[ 9] - mIn.f[ 8] * mIn.f[ 1] ) * (float)det_1;
        mDummyMatrix.f[10] =   ( mIn.f[ 0] * mIn.f[ 5] - mIn.f[ 4] * mIn.f[ 1] ) * (float)det_1;

        /* Calculate -C * inverse(A) */
        mDummyMatrix.f[12] = - ( mIn.f[12] * mDummyMatrix.f[ 0] + mIn.f[13] * mDummyMatrix.f[ 4] + mIn.f[14] * mDummyMatrix.f[ 8] );
        mDummyMatrix.f[13] = - ( mIn.f[12] * mDummyMatrix.f[ 1] + mIn.f[13] * mDummyMatrix.f[ 5] + mIn.f[14] * mDummyMatrix.f[ 9] );
        mDummyMatrix.f[14] = - ( mIn.f[12] * mDummyMatrix.f[ 2] + mIn.f[13] * mDummyMatrix.f[ 6] + mIn.f[14] * mDummyMatrix.f[10] );

        /* Fill in last row */
        mDummyMatrix.f[ 3] = 0.0f;
		mDummyMatrix.f[ 7] = 0.0f;
		mDummyMatrix.f[11] = 0.0f;
        mDummyMatrix.f[15] = 1.0f;
	}

   	/* Copy contents of dummy matrix in pfMatrix */
	mOut = mDummyMatrix;
}



void MatrixInverseEx(
	MATRIX			&mOut,
	const MATRIX	&mIn)
{
	MATRIX		mTmp;
	float 			*ppfRows[4];
	float 			pfRes[4];
	float 			pfIn[20];
	int				i, j;

	for(i = 0; i < 4; ++i)
		ppfRows[i] = &pfIn[i * 5];

	/* Solve 4 sets of 4 linear equations */
	for(i = 0; i < 4; ++i)
	{
		for(j = 0; j < 4; ++j)
		{
			ppfRows[j][0] = c_mIdentity.f[i + 4 * j];
			memcpy(&ppfRows[j][1], &mIn.f[j * 4], 4 * sizeof(float));
		}

		MatrixLinearEqSolve(pfRes, (float**)ppfRows, 4);

		for(j = 0; j < 4; ++j)
		{
			mTmp.f[i + 4 * j] = pfRes[j];
		}
	}

	mOut = mTmp;
}


void MatrixLookAtLH(
	MATRIX			&mOut,
	const VECTOR3	&vEye,
	const VECTOR3	&vAt,
	const VECTOR3	&vUp)
{
	VECTOR3 f, vUpActual, s, u;
	MATRIX	t;

	f.x = vEye.x - vAt.x;
	f.y = vEye.y - vAt.y;
	f.z = vEye.z - vAt.z;

	MatrixVec3Normalize(f, f);
	MatrixVec3Normalize(vUpActual, vUp);
	MatrixVec3CrossProduct(s, f, vUpActual);
	MatrixVec3CrossProduct(u, s, f);

	mOut.f[ 0] = s.x;
	mOut.f[ 1] = u.x;
	mOut.f[ 2] = -f.x;
	mOut.f[ 3] = 0;

	mOut.f[ 4] = s.y;
	mOut.f[ 5] = u.y;
	mOut.f[ 6] = -f.y;
	mOut.f[ 7] = 0;

	mOut.f[ 8] = s.z;
	mOut.f[ 9] = u.z;
	mOut.f[10] = -f.z;
	mOut.f[11] = 0;

	mOut.f[12] = 0;
	mOut.f[13] = 0;
	mOut.f[14] = 0;
	mOut.f[15] = 1;

	MatrixTranslation(t, -vEye.x, -vEye.y, -vEye.z);
	MatrixMultiply(mOut, t, mOut);
}


void MatrixLookAtRH(
	MATRIX			&mOut,
	const VECTOR3	&vEye,
	const VECTOR3	&vAt,
	const VECTOR3	&vUp)
{
	VECTOR3 f, vUpActual, s, u;
	MATRIX	t;

	f.x = vAt.x - vEye.x;
	f.y = vAt.y - vEye.y;
	f.z = vAt.z - vEye.z;

	MatrixVec3Normalize(f, f);
	MatrixVec3Normalize(vUpActual, vUp);
	MatrixVec3CrossProduct(s, f, vUpActual);
	MatrixVec3CrossProduct(u, s, f);

	mOut.f[ 0] = s.x;
	mOut.f[ 1] = u.x;
	mOut.f[ 2] = -f.x;
	mOut.f[ 3] = 0;

	mOut.f[ 4] = s.y;
	mOut.f[ 5] = u.y;
	mOut.f[ 6] = -f.y;
	mOut.f[ 7] = 0;

	mOut.f[ 8] = s.z;
	mOut.f[ 9] = u.z;
	mOut.f[10] = -f.z;
	mOut.f[11] = 0;

	mOut.f[12] = 0;
	mOut.f[13] = 0;
	mOut.f[14] = 0;
	mOut.f[15] = 1;

	MatrixTranslation(t, -vEye.x, -vEye.y, -vEye.z);
	MatrixMultiply(mOut, t, mOut);
}


void MatrixPerspectiveFovLH(
	MATRIX	&mOut,
	const float	fFOVy,
	const float	fAspect,
	const float	fNear,
	const float	fFar,
	const bool  bRotate)
{
	float f, n, fRealAspect;

	if (bRotate)
		fRealAspect = 1.0f / fAspect;
	else
		fRealAspect = fAspect;

	// cotangent(a) == 1.0f / tan(a);
	f = 1.0f / (float)tan(fFOVy * 0.5f);
	n = 1.0f / (fFar - fNear);

	mOut.f[ 0] = f / fRealAspect;
	mOut.f[ 1] = 0;
	mOut.f[ 2] = 0;
	mOut.f[ 3] = 0;

	mOut.f[ 4] = 0;
	mOut.f[ 5] = f;
	mOut.f[ 6] = 0;
	mOut.f[ 7] = 0;

	mOut.f[ 8] = 0;
	mOut.f[ 9] = 0;
	mOut.f[10] = fFar * n;
	mOut.f[11] = 1;

	mOut.f[12] = 0;
	mOut.f[13] = 0;
	mOut.f[14] = -fFar * fNear * n;
	mOut.f[15] = 0;

	if (bRotate)
	{
		MATRIX mRotation, mTemp = mOut;
		MatrixRotationZ(mRotation, 90.0f*PIf/180.0f);
		MatrixMultiply(mOut, mTemp, mRotation);
	}
}


void MatrixPerspectiveFovRH(
	MATRIX	&mOut,
	const float	fFOVy,
	const float	fAspect,
	const float	fNear,
	const float	fFar,
	const bool  bRotate)
{
	float f, n, fRealAspect;

	if (bRotate)
		fRealAspect = 1.0f / fAspect;
	else
		fRealAspect = fAspect;

	// cotangent(a) == 1.0f / tan(a);
	f = 1.0f / (float)tan(fFOVy * 0.5f);
	n = 1.0f / (fNear - fFar);

	mOut.f[ 0] = f / fRealAspect;
	mOut.f[ 1] = 0;
	mOut.f[ 2] = 0;
	mOut.f[ 3] = 0;

	mOut.f[ 4] = 0;
	mOut.f[ 5] = f;
	mOut.f[ 6] = 0;
	mOut.f[ 7] = 0;

	mOut.f[ 8] = 0;
	mOut.f[ 9] = 0;
	mOut.f[10] = (fFar + fNear) * n;
	mOut.f[11] = -1;

	mOut.f[12] = 0;
	mOut.f[13] = 0;
	mOut.f[14] = (2 * fFar * fNear) * n;
	mOut.f[15] = 0;

	if (bRotate)
	{
		MATRIX mRotation, mTemp = mOut;
		MatrixRotationZ(mRotation, 90.0f*PIf/180.0f);
		MatrixMultiply(mOut, mTemp, mRotation);
	}
}


void MatrixOrthoLH(
	MATRIX	&mOut,
	const float w,
	const float h,
	const float zn,
	const float zf,
	const bool  bRotate)
{
	mOut.f[ 0] = 2 / w;
	mOut.f[ 1] = 0;
	mOut.f[ 2] = 0;
	mOut.f[ 3] = 0;

	mOut.f[ 4] = 0;
	mOut.f[ 5] = 2 / h;
	mOut.f[ 6] = 0;
	mOut.f[ 7] = 0;

	mOut.f[ 8] = 0;
	mOut.f[ 9] = 0;
	mOut.f[10] = 1 / (zf - zn);
	mOut.f[11] = zn / (zn - zf);

	mOut.f[12] = 0;
	mOut.f[13] = 0;
	mOut.f[14] = 0;
	mOut.f[15] = 1;

	if (bRotate)
	{
		MATRIX mRotation, mTemp = mOut;
		MatrixRotationZ(mRotation, -90.0f*PIf/180.0f);
		MatrixMultiply(mOut, mRotation, mTemp);
	}
}


void MatrixOrthoRH(
	MATRIX	&mOut,
	const float w,
	const float h,
	const float zn,
	const float zf,
	const bool  bRotate)
{
	mOut.f[ 0] = 2 / w;
	mOut.f[ 1] = 0;
	mOut.f[ 2] = 0;
	mOut.f[ 3] = 0;

	mOut.f[ 4] = 0;
	mOut.f[ 5] = 2 / h;
	mOut.f[ 6] = 0;
	mOut.f[ 7] = 0;

	mOut.f[ 8] = 0;
	mOut.f[ 9] = 0;
	mOut.f[10] = 1 / (zn - zf);
	mOut.f[11] = zn / (zn - zf);

	mOut.f[12] = 0;
	mOut.f[13] = 0;
	mOut.f[14] = 0;
	mOut.f[15] = 1;

	if (bRotate)
	{
		MATRIX mRotation, mTemp = mOut;
		MatrixRotationZ(mRotation, -90.0f*PIf/180.0f);
		MatrixMultiply(mOut, mRotation, mTemp);
	}
}


void MatrixVec3Lerp(
	VECTOR3		&vOut,
	const VECTOR3	&v1,
	const VECTOR3	&v2,
	const float	s)
{
	vOut.x = v1.x + s * (v2.x - v1.x);
	vOut.y = v1.y + s * (v2.y - v1.y);
	vOut.z = v1.z + s * (v2.z - v1.z);
}


float MatrixVec3DotProduct(
	const VECTOR3	&v1,
	const VECTOR3	&v2)
{
	return (v1.x*v2.x + v1.y*v2.y + v1.z*v2.z);
}

void MatrixVec3Multiply(VECTOR3		&vOut,
						const VECTOR3	&vIn,
						const MATRIX	&mIn)
{
	    VECTOR3 result;
	
		/* Perform calculation on a dummy VECTOR (result) */
		result.x = mIn.f[_11] * vIn.x + mIn.f[_21] * vIn.y + mIn.f[_31] * vIn.z;
		result.y = mIn.f[_12] * vIn.x + mIn.f[_22] * vIn.y + mIn.f[_32] * vIn.z;
		result.z = mIn.f[_13] * vIn.x + mIn.f[_23] * vIn.y + mIn.f[_33] * vIn.z;
	    
	    vOut = result;
}



void MatrixVec3CrossProduct(
	VECTOR3		&vOut,
	const VECTOR3	&v1,
	const VECTOR3	&v2)
{
    VECTOR3 result;

	/* Perform calculation on a dummy VECTOR (result) */
    result.x = v1.y * v2.z - v1.z * v2.y;
    result.y = v1.z * v2.x - v1.x * v2.z;
    result.z = v1.x * v2.y - v1.y * v2.x;

	/* Copy result in pOut */
	vOut = result;
}


void MatrixVec3Normalize(
	VECTOR3		&vOut,
	const VECTOR3	&vIn)
{
	float	f;
	double temp;

	temp = (double)(vIn.x * vIn.x + vIn.y * vIn.y + vIn.z * vIn.z);
	temp = 1.0 / sqrt(temp);
	f = (float)temp;

	vOut.x = vIn.x * f;
	vOut.y = vIn.y * f;
	vOut.z = vIn.z * f;
}

void MatrixVec4Normalize(
	VECTOR4		&vOut,
	const VECTOR4	&vIn)
{
	float	f;
	double temp;

	temp = (double)(vIn.x * vIn.x + vIn.y * vIn.y + vIn.z * vIn.z);
	temp = 1.0 / sqrt(temp);
	f = (float)temp;

	vOut.x = vIn.x * f;
	vOut.y = vIn.y * f;
	vOut.z = vIn.z * f;
	vOut.w = vIn.w * f;
}



float MatrixVec3Length(
	const VECTOR3	&vIn)
{
	double temp;

	temp = (double)(vIn.x * vIn.x + vIn.y * vIn.y + vIn.z * vIn.z);
	return (float) sqrt(temp);
}



void MatrixQuaternionIdentity(
	QUATERNION		&qOut)
{
	qOut.x = 0;
	qOut.y = 0;
	qOut.z = 0;
	qOut.w = 1;
}


void MatrixQuaternionRotationAxis(
	QUATERNION		&qOut,
	const VECTOR3	&vAxis,
	const float			fAngle)
{
	float	fSin, fCos;

	fSin = (float)sin(fAngle * 0.5f);
	fCos = (float)cos(fAngle * 0.5f);

	/* Create quaternion */
	qOut.x = vAxis.x * fSin;
	qOut.y = vAxis.y * fSin;
	qOut.z = vAxis.z * fSin;
	qOut.w = fCos;

	/* Normalise it */
	MatrixQuaternionNormalize(qOut);
}


void MatrixQuaternionToAxisAngle(
	const QUATERNION	&qIn,
	VECTOR3			&vAxis,
	float					&fAngle)
{
	float	fCosAngle, fSinAngle;
	double	temp;

	/* Compute some values */
	fCosAngle	= qIn.w;
	temp		= 1.0f - fCosAngle*fCosAngle;
	fAngle		= (float)acos(fCosAngle)*2.0f;
	fSinAngle	= (float)sqrt(temp);

	/* This is to avoid a division by zero */
	if ((float)fabs(fSinAngle)<0.0005f)
		fSinAngle = 1.0f;

	/* Get axis vector */
	vAxis.x = qIn.x / fSinAngle;
	vAxis.y = qIn.y / fSinAngle;
	vAxis.z = qIn.z / fSinAngle;
}

void MatrixQuaternionSlerp(
	QUATERNION			&qOut,
	const QUATERNION	&qA,
	const QUATERNION	&qB,
	const float				t)
{
	float		fCosine, fAngle, A, B;

	/* Parameter checking */
	if (t<0.0f || t>1.0f)
	{
		printf("MatrixQuaternionSlerp : Bad parameters\n");
		qOut.x = 0;
		qOut.y = 0;
		qOut.z = 0;
		qOut.w = 1;
		return;
	}

	/* Find sine of Angle between Quaternion A and B (dot product between quaternion A and B) */
	fCosine = qA.w*qB.w + qA.x*qB.x + qA.y*qB.y + qA.z*qB.z;

	if (fCosine < 0)
	{
		QUATERNION qi;

		/*
			<http://www.magic-software.com/Documentation/Quaternions.pdf>

			"It is important to note that the quaternions q and -q represent
			the same rotation... while either quaternion will do, the
			interpolation methods require choosing one over the other.

			"Although q1 and -q1 represent the same rotation, the values of
			Slerp(t; q0, q1) and Slerp(t; q0,-q1) are not the same. It is
			customary to choose the sign... on q1 so that... the angle
			between q0 and q1 is acute. This choice avoids extra
			spinning caused by the interpolated rotations."
		*/
		qi.x = -qB.x;
		qi.y = -qB.y;
		qi.z = -qB.z;
		qi.w = -qB.w;

		MatrixQuaternionSlerp(qOut, qA, qi, t);
		return;
	}

	fCosine = _MIN(fCosine, 1.0f);
	fAngle = (float)acos(fCosine);

	/* Avoid a division by zero */
	if (fAngle==0.0f)
	{
		qOut = qA;
		return;
	}

	/* Precompute some values */
	A = (float)(sin((1.0f-t)*fAngle) / sin(fAngle));
	B = (float)(sin(t*fAngle) / sin(fAngle));

	/* Compute resulting quaternion */
	qOut.x = A * qA.x + B * qB.x;
	qOut.y = A * qA.y + B * qB.y;
	qOut.z = A * qA.z + B * qB.z;
	qOut.w = A * qA.w + B * qB.w;

	/* Normalise result */
	MatrixQuaternionNormalize(qOut);
}



void MatrixQuaternionNormalize(QUATERNION &quat)
{
	float	fMagnitude;
	double	temp;

	/* Compute quaternion magnitude */
	temp = quat.w*quat.w + quat.x*quat.x + quat.y*quat.y + quat.z*quat.z;
	fMagnitude = (float)sqrt(temp);

	/* Divide each quaternion component by this magnitude */
	if (fMagnitude!=0.0f)
	{
		fMagnitude = 1.0f / fMagnitude;
		quat.x *= fMagnitude;
		quat.y *= fMagnitude;
		quat.z *= fMagnitude;
		quat.w *= fMagnitude;
	}
}


void MatrixRotationQuaternion(
	MATRIX				&mOut,
	const QUATERNION	&quat)
{
	const QUATERNION *pQ;

#if defined(BUILD_DX9) || defined(BUILD_D3DM)
	QUATERNION qInv;

	qInv.x = -quat.x;
	qInv.y = -quat.y;
	qInv.z = -quat.z;
	qInv.w = quat.w;

	pQ = &qInv;
#else
	pQ = &quat;
#endif

    /* Fill matrix members */
	mOut.f[0] = 1.0f - 2.0f*pQ->y*pQ->y - 2.0f*pQ->z*pQ->z;
	mOut.f[1] = 2.0f*pQ->x*pQ->y - 2.0f*pQ->z*pQ->w;
	mOut.f[2] = 2.0f*pQ->x*pQ->z + 2.0f*pQ->y*pQ->w;
	mOut.f[3] = 0.0f;

	mOut.f[4] = 2.0f*pQ->x*pQ->y + 2.0f*pQ->z*pQ->w;
	mOut.f[5] = 1.0f - 2.0f*pQ->x*pQ->x - 2.0f*pQ->z*pQ->z;
	mOut.f[6] = 2.0f*pQ->y*pQ->z - 2.0f*pQ->x*pQ->w;
	mOut.f[7] = 0.0f;

	mOut.f[8] = 2.0f*pQ->x*pQ->z - 2*pQ->y*pQ->w;
	mOut.f[9] = 2.0f*pQ->y*pQ->z + 2.0f*pQ->x*pQ->w;
	mOut.f[10] = 1.0f - 2.0f*pQ->x*pQ->x - 2*pQ->y*pQ->y;
	mOut.f[11] = 0.0f;

	mOut.f[12] = 0.0f;
	mOut.f[13] = 0.0f;
	mOut.f[14] = 0.0f;
	mOut.f[15] = 1.0f;
}


void MatrixQuaternionMultiply(
	QUATERNION			&qOut,
	const QUATERNION	&qA,
	const QUATERNION	&qB)
{
	VECTOR3	CrossProduct;

	/* Compute scalar component */
	qOut.w = (qA.w*qB.w) - (qA.x*qB.x + qA.y*qB.y + qA.z*qB.z);

	/* Compute cross product */
	CrossProduct.x = qA.y*qB.z - qA.z*qB.y;
	CrossProduct.y = qA.z*qB.x - qA.x*qB.z;
	CrossProduct.z = qA.x*qB.y - qA.y*qB.x;

	/* Compute result vector */
	qOut.x = (qA.w * qB.x) + (qB.w * qA.x) + CrossProduct.x;
	qOut.y = (qA.w * qB.y) + (qB.w * qA.y) + CrossProduct.y;
	qOut.z = (qA.w * qB.z) + (qB.w * qA.z) + CrossProduct.z;

	/* Normalize resulting quaternion */
	MatrixQuaternionNormalize(qOut);
}


void MatrixLinearEqSolve(
	float		* const pRes,
	float		** const pSrc,	// 2D array of floats. 4 Eq linear problem is 5x4 matrix, constants in first column.
	const int	nCnt)
{
	int		i, j, k;
	float	f;

#if 0
	/*
		Show the matrix in debug output
	*/
	_RPT1(_CRT_WARN, "LinearEqSolve(%d)\n", nCnt);
	for(i = 0; i < nCnt; ++i)
	{
		_RPT1(_CRT_WARN, "%.8f |", pSrc[i][0]);
		for(j = 1; j <= nCnt; ++j)
			_RPT1(_CRT_WARN, " %.8f", pSrc[i][j]);
		_RPT0(_CRT_WARN, "\n");
	}
#endif

	if(nCnt == 1)
	{
		_ASSERT(pSrc[0][1] != 0);
		pRes[0] = pSrc[0][0] / pSrc[0][1];
		return;
	}

	// Loop backwards in an attempt avoid the need to swap rows
	i = nCnt;
	while(i)
	{
		--i;

		if(pSrc[i][nCnt] != 0)
		{
			// Row i can be used to zero the other rows; let's move it to the bottom
			if(i != (nCnt-1))
			{
				for(j = 0; j <= nCnt; ++j)
				{
					// Swap the two values
					f = pSrc[nCnt-1][j];
					pSrc[nCnt-1][j] = pSrc[i][j];
					pSrc[i][j] = f;
				}
			}

			// Now zero the last columns of the top rows
			for(j = 0; j < (nCnt-1); ++j)
			{
				_ASSERT(pSrc[nCnt-1][nCnt] != 0);
				f = pSrc[j][nCnt] / pSrc[nCnt-1][nCnt];

				// No need to actually calculate a zero for the final column
				for(k = 0; k < nCnt; ++k)
				{
					pSrc[j][k] -= f * pSrc[nCnt-1][k];
				}
			}

			break;
		}
	}

	// Solve the top-left sub matrix
	MatrixLinearEqSolve(pRes, pSrc, nCnt - 1);

	// Now calc the solution for the bottom row
	f = pSrc[nCnt-1][0];
	for(k = 1; k < nCnt; ++k)
	{
		f -= pSrc[nCnt-1][k] * pRes[k-1];
	}
	_ASSERT(pSrc[nCnt-1][nCnt] != 0);
	f /= pSrc[nCnt-1][nCnt];
	pRes[nCnt-1] = f;

#if 0
	{
		float fCnt;

		/*
			Verify that the result is correct
		*/
		fCnt = 0;
		for(i = 1; i <= nCnt; ++i)
			fCnt += pSrc[nCnt-1][i] * pRes[i-1];

		_ASSERT(_ABS(fCnt - pSrc[nCnt-1][0]) < 1e-3);
	}
#endif
}


