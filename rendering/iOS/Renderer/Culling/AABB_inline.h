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
#ifndef AABB_INLINE_H_
#define AABB_INLINE_H_

#include "Mathematics.h"
#include "Macros.h"

#include "AABB.h"

inline void CAABB::ComputeAABB(const VECTOR3* const pV, const int nNumberOfVertices)
{
	int			i;
//	VERTTYPE	MinX, MaxX, MinY, MaxY, MinZ, MaxZ;
	VECTOR3 min; 
	VECTOR3 max;

	// Inialise values to first vertex //
	min.x=pV->x;	max.x=pV->x;
	min.y=pV->y;	max.y=pV->y;
	min.z=pV->z;	max.z=pV->z;

	// Loop through all vertices to find extremas //
	for (i=1; i < nNumberOfVertices; i++)
	{
		// Minimum and Maximum X //
		if (pV[i].x < min.x) min.x = pV[i].x;
		if (pV[i].x > max.x) max.x = pV[i].x;

		// Minimum and Maximum Y //
		if (pV[i].y < min.y) min.y = pV[i].y;
		if (pV[i].y > max.y) max.y = pV[i].y;

		// Minimum and Maximum Z //
		if (pV[i].z < min.z) min.z = pV[i].z;
		if (pV[i].z > max.z) max.z = pV[i].z;
	}
	
	Set(min, max);
}

inline void CAABB::ComputeAABB(const float	* pV, const int stride, const int nNumberOfVertices)
{
	int			i;
	//	VERTTYPE	MinX, MaxX, MinY, MaxY, MinZ, MaxZ;
	VECTOR3 min; 
	VECTOR3 max;
	
	// Inialise values to first vertex //
	min.x=pV[0];	max.x=pV[0];
	min.y=pV[1];	max.y=pV[1];
	min.z=pV[2];	max.z=pV[2];
	
	// Loop through all vertices to find extremas //
	for (i=1; i < nNumberOfVertices; i++)
	{
		// Minimum and Maximum X //
		if (pV[0] < min.x) min.x = pV[0];
		if (pV[0] > max.x) max.x = pV[0];
		
		// Minimum and Maximum Y //
		if (pV[1] < min.y) min.y = pV[1];
		if (pV[1] > max.y) max.y = pV[1];
		
		// Minimum and Maximum Z //
		if (pV[2] < min.z) min.z = pV[2];
		if (pV[2] > max.z) max.z = pV[2];
			
		pV = (float *)(((int)pV)+stride);
	}
	
	Set(min, max);
}

inline void CAABB::Set(const VECTOR3& min, const VECTOR3& max) 
{
	
	VECTOR3 Temp;
	Temp.x = min.x - max.x;
	Temp.y = min.y - max.y;
	Temp.z = min.z - max.z;
	
	//float Mag = sqrtf(Temp.x * Temp.x + Temp.y * Temp.y + Temp.z * Temp.z);
	VERTTYPE Mag = MatrixVec3Length(Temp);
		
	VERTTYPE BoundingRadius = Mag * 0.5f;
	Set(min, max, BoundingRadius);
}

inline void CAABB::Set(const VECTOR3& min, const VECTOR3& max, float BoundingRadius)
{
	
	m_Min = min; 
	m_Max = max; 
	m_BoundingRadius = BoundingRadius;
}

inline VECTOR3 CAABB::GetCenter(void) const
{
	VECTOR3 Center;
	// XDIV for integer?
	Center.x = (m_Min.x + m_Max.x) / f2vt(2.0f);
	Center.y = (m_Min.y + m_Max.y) / f2vt(2.0f);
	Center.z = (m_Min.z + m_Max.z) / f2vt(2.0f);

	return Center;
}

inline VERTTYPE CAABB::GetBoundingRadius(void) const {return m_BoundingRadius;}

//
// transform the minimum and maximum corners of this box with a matrix
// and set this AABB up to span the new transformed corners
//
inline void CAABB::Transform(const MATRIX &matrix)
{
	VECTOR3 box[8];
	VECTOR3 finalMin, finalMax;
	VECTOR4 transformed;

	box[0].x = m_Min.x;
	box[0].y = m_Min.y;
	box[0].z = m_Min.z;
	
	box[1].x = m_Min.x;
	box[1].y = m_Min.y;
	box[1].z = m_Max.z;
	
	box[2].x = m_Min.x;
	box[2].y = m_Max.y;
	box[2].z = m_Min.z;
	
	box[3].x = m_Min.x;
	box[3].y = m_Max.y;
	box[3].z = m_Max.z;
	
	box[4].x = m_Max.x;
	box[4].y = m_Min.y;
	box[4].z = m_Min.z;
	
	box[5].x = m_Max.x;
	box[5].y = m_Min.y;
	box[5].z = m_Max.z;
	
	box[6].x = m_Max.x;
	box[6].y = m_Max.y;
	box[6].z = m_Min.z;
	
	box[7].x = m_Max.x;
	box[7].y = m_Max.y;
	box[7].z = m_Max.z;
	
	TransTransform((VECTOR4 *) &finalMin, (const VECTOR4 *) &box[0], &matrix);
	finalMax = finalMin;
	
	for(int x = 1; x < 8; x++)
	{
		TransTransform(&transformed, (const VECTOR4 *) &box[x], &matrix);
		finalMin.x = _MIN(finalMin.x, transformed.x);
		finalMin.y = _MIN(finalMin.y, transformed.y);
		finalMin.z = _MIN(finalMin.z, transformed.z);
		finalMax.x = _MAX(finalMax.x, transformed.x);
		finalMax.y = _MAX(finalMax.y, transformed.y);
		finalMax.z = _MAX(finalMax.z, transformed.z);
	}
	
	Set(finalMin, finalMax);
}


inline bool CAABB::Contains(VECTOR3 vec) const
{
	VECTOR3 result1;
	result1.x = ((m_Min.x <= vec.x) ?  f2vt(1.0f) :  f2vt(0.0f));
	result1.y = ((m_Min.y <= vec.y) ?  f2vt(1.0f) :  f2vt(0.0f));
	result1.z = ((m_Min.z <= vec.z) ?  f2vt(1.0f) :  f2vt(0.0f));
	
	VECTOR3 result2;
	result2.x = ((m_Max.x >= vec.x) ?  f2vt(1.0f) :  f2vt(0.0f));
	result2.y = ((m_Max.y >= vec.y) ?  f2vt(1.0f) :  f2vt(0.0f));
	result2.z = ((m_Max.z >= vec.z) ?  f2vt(1.0f) :  f2vt(0.0f));
	
	// don't know if that really works :-)
	if(MatrixVec3DotProduct(result1, result2) == f2vt(3.0f))
		return true;
	else
		return false;
}



inline const VECTOR3 &CAABB::GetMin(void) const {return m_Min;}
inline const VECTOR3 &CAABB::GetMax(void) const {return m_Max;}
inline VERTTYPE CAABB::GetWidth(void) const {return _ABS(m_Max.x - m_Min.x);}
inline VERTTYPE CAABB::GetHeight(void) const {return _ABS(m_Max.y - m_Min.y);}
inline VERTTYPE CAABB::GetDepth(void) const {return _ABS(m_Max.z - m_Min.z);}

inline bool CAABB::IsVisible( const MATRIX	* const pMatrix,
							  bool			* const pNeedsZClipping)
{
	VERTTYPE	fX, fY, fZ, fW;
	int			i, nX0, nX1, nY0, nY1, nZ;
	
	nX0 = 8;
	nX1 = 8;
	nY0 = 8;
	nY1 = 8;
	nZ  = 8;
	
	VECTOR3 Point[8];
	Point[0].x = m_Min.x; Point[0].y = m_Min.y; Point[0].z = m_Min.z;
	Point[1].x = m_Min.x; Point[1].y = m_Min.y; Point[1].z = m_Max.z;
	Point[2].x = m_Min.x; Point[2].y = m_Max.y; Point[2].z = m_Min.z;
	Point[3].x = m_Min.x; Point[3].y = m_Max.y; Point[3].z = m_Max.z;
	Point[4].x = m_Max.x; Point[4].y = m_Min.y; Point[4].z = m_Min.z;
	Point[5].x = m_Max.x; Point[5].y = m_Min.y; Point[5].z = m_Max.z;
	Point[6].x = m_Max.x; Point[6].y = m_Max.y; Point[6].z = m_Min.z;
	Point[7].x = m_Max.x; Point[7].y = m_Max.y; Point[7].z = m_Max.z;
	
	// Transform the eight bounding box vertices //
	i = 8;
	while(i)
	{
		i--;
		fX =	pMatrix->f[ 0]*Point[i].x +
				pMatrix->f[ 4]*Point[i].y +
				pMatrix->f[ 8]*Point[i].z +
				pMatrix->f[12];
		fY =	pMatrix->f[ 1]*Point[i].x +
				pMatrix->f[ 5]*Point[i].y +
				pMatrix->f[ 9]*Point[i].z +
				pMatrix->f[13];
		fZ =	pMatrix->f[ 2]*Point[i].x +
				pMatrix->f[ 6]*Point[i].y +
				pMatrix->f[10]*Point[i].z +
				pMatrix->f[14];
		fW =	pMatrix->f[ 3]*Point[i].x +
				pMatrix->f[ 7]*Point[i].y +
				pMatrix->f[11]*Point[i].z +
				pMatrix->f[15];
		
		if(fX < -fW)
			nX0--;
		else if(fX > fW)
			nX1--;
		
		if(fY < -fW)
			nY0--;
		else if(fY > fW)
			nY1--;
		
		if(fZ < 0)
			nZ--;
	}
	
	if(nZ)
	{
		if(!(nX0 * nX1 * nY0 * nY1))
		{
			*pNeedsZClipping = false;
			return false;
		}
		
		if(nZ == 8)
		{
			*pNeedsZClipping = false;
			return true;
		}
		
		*pNeedsZClipping = true;
		return true;
	}
	else
	{
		*pNeedsZClipping = false;
		return false;
	}
}


/*
typedef struct BOUNDINGBOX_TAG
{
	VECTOR3	Point[8];
} BOUNDINGBOX, *LPBOUNDINGBOX;

//
// @Function			TransComputeBoundingBox
// @Output			pBoundingBox
// @Input				pV
// @Input				nNumberOfVertices
// @Description		Calculate the eight vertices that surround an object.
//					This "bounding box" is used later to determine whether
//					the object is visible or not.
//					This function should only be called once to determine the
//					object's bounding box.
//////////////////////////////////////////////////////////////////////////////
void TransComputeBoundingBox(
	BOUNDINGBOX		* const pBoundingBox,
	const VECTOR3	* const pV,
	const int			nNumberOfVertices);

//!//////////////////////////////////////////////////////////////////////////////
// @Function			TransIsBoundingBoxVisible
// @Output			pNeedsZClipping
// @Input				pBoundingBox
// @Input				pMatrix
// @Return			TRUE if the object is visible, FALSE if not.
// @Description		Determine if a bounding box is "visible" or not along the
//					Z axis.
//					If the function returns TRUE, the object is visible and should
//					be displayed (check bNeedsZClipping to know if Z Clipping needs
//					to be done).
//					If the function returns FALSE, the object is not visible and thus
//					does not require to be displayed.
//					bNeedsZClipping indicates whether the object needs Z Clipping
//					(i.e. the object is partially visible).
//					- *pBoundingBox is a pointer to the bounding box structure.
//					- *pMatrix is the World, View & Projection matrices combined.
//					- *bNeedsZClipping is TRUE if Z clipping is required.
//////////////////////////////////////////////////////////////////////////////
bool TransIsBoundingBoxVisible(
	const BOUNDINGBOX	* const pBoundingBox,
	const MATRIX		* const pMatrix,
	bool					* const pNeedsZClipping);

//!///////////////////////////////////////////////////////////////////////////
// @Function			TransComputeBoundingBox
// @Output			pBoundingBox
// @Input				pV
// @Input				nNumberOfVertices
// @Description		Calculate the eight vertices that surround an object.
//					This "bounding box" is used later to determine whether
//					the object is visible or not.
//					This function should only be called once to determine the
//					object's bounding box.
//////////////////////////////////////////////////////////////////////////////
void TransComputeBoundingBox(
	BOUNDINGBOX		* const pBoundingBox,
	const VECTOR3	* const pV,
	const int			nNumberOfVertices)
{
	int			i;
	VERTTYPE	MinX, MaxX, MinY, MaxY, MinZ, MaxZ;

	// Inialise values to first vertex //
	MinX=pV->x;	MaxX=pV->x;
	MinY=pV->y;	MaxY=pV->y;
	MinZ=pV->z;	MaxZ=pV->z;

	// Loop through all vertices to find extremas //
	for (i=1; i<nNumberOfVertices; i++)
	{
		// Minimum and Maximum X //
		if (pV[i].x < MinX) MinX = pV[i].x;
		if (pV[i].x > MaxX) MaxX = pV[i].x;

		// Minimum and Maximum Y //
		if (pV[i].y < MinY) MinY = pV[i].y;
		if (pV[i].y > MaxY) MaxY = pV[i].y;

		// Minimum and Maximum Z //
		if (pV[i].z < MinZ) MinZ = pV[i].z;
		if (pV[i].z > MaxZ) MaxZ = pV[i].z;
	}

	// Assign the resulting extremas to the bounding box structure //
	// Point 0 //
	pBoundingBox->Point[0].x=MinX;
	pBoundingBox->Point[0].y=MinY;
	pBoundingBox->Point[0].z=MinZ;

	// Point 1 //
	pBoundingBox->Point[1].x=MinX;
	pBoundingBox->Point[1].y=MinY;
	pBoundingBox->Point[1].z=MaxZ;

	// Point 2 //
	pBoundingBox->Point[2].x=MinX;
	pBoundingBox->Point[2].y=MaxY;
	pBoundingBox->Point[2].z=MinZ;

	// Point 3 //
	pBoundingBox->Point[3].x=MinX;
	pBoundingBox->Point[3].y=MaxY;
	pBoundingBox->Point[3].z=MaxZ;

	// Point 4 //
	pBoundingBox->Point[4].x=MaxX;
	pBoundingBox->Point[4].y=MinY;
	pBoundingBox->Point[4].z=MinZ;

	// Point 5 //
	pBoundingBox->Point[5].x=MaxX;
	pBoundingBox->Point[5].y=MinY;
	pBoundingBox->Point[5].z=MaxZ;

	// Point 6 //
	pBoundingBox->Point[6].x=MaxX;
	pBoundingBox->Point[6].y=MaxY;
	pBoundingBox->Point[6].z=MinZ;

	// Point 7 //
	pBoundingBox->Point[7].x=MaxX;
	pBoundingBox->Point[7].y=MaxY;
	pBoundingBox->Point[7].z=MaxZ;
}


//!//////////////////////////////////////////////////////////////////////////////
// @Function			TransIsBoundingBoxVisible
// @Output			pNeedsZClipping
// @Input				pBoundingBox
// @Input				pMatrix
// @Return			TRUE if the object is visible, FALSE if not.
// @Description		Determine if a bounding box is "visible" or not along the
//					Z axis.
//					If the function returns TRUE, the object is visible and should
//					be displayed (check bNeedsZClipping to know if Z Clipping needs
//					to be done).
//					If the function returns FALSE, the object is not visible and thus
//					does not require to be displayed.
//					bNeedsZClipping indicates whether the object needs Z Clipping
//					(i.e. the object is partially visible).
//					- *pBoundingBox is a pointer to the bounding box structure.
//					- *pMatrix is the World, View & Projection matrices combined.
//					- *bNeedsZClipping is TRUE if Z clipping is required.
//////////////////////////////////////////////////////////////////////////////
bool TransIsBoundingBoxVisible(
	const BOUNDINGBOX	* const pBoundingBox,
	const MATRIX		* const pMatrix,
	bool					* const pNeedsZClipping)
{
	VERTTYPE	fX, fY, fZ, fW;
	int			i, nX0, nX1, nY0, nY1, nZ;

	nX0 = 8;
	nX1 = 8;
	nY0 = 8;
	nY1 = 8;
	nZ  = 8;

	// Transform the eight bounding box vertices //
	i = 8;
	while(i)
	{
		i--;
		fX =	pMatrix->f[ 0]*pBoundingBox->Point[i].x +
				pMatrix->f[ 4]*pBoundingBox->Point[i].y +
				pMatrix->f[ 8]*pBoundingBox->Point[i].z +
				pMatrix->f[12];
		fY =	pMatrix->f[ 1]*pBoundingBox->Point[i].x +
				pMatrix->f[ 5]*pBoundingBox->Point[i].y +
				pMatrix->f[ 9]*pBoundingBox->Point[i].z +
				pMatrix->f[13];
		fZ =	pMatrix->f[ 2]*pBoundingBox->Point[i].x +
				pMatrix->f[ 6]*pBoundingBox->Point[i].y +
				pMatrix->f[10]*pBoundingBox->Point[i].z +
				pMatrix->f[14];
		fW =	pMatrix->f[ 3]*pBoundingBox->Point[i].x +
				pMatrix->f[ 7]*pBoundingBox->Point[i].y +
				pMatrix->f[11]*pBoundingBox->Point[i].z +
				pMatrix->f[15];

		if(fX < -fW)
			nX0--;
		else if(fX > fW)
			nX1--;

		if(fY < -fW)
			nY0--;
		else if(fY > fW)
			nY1--;

		if(fZ < 0)
			nZ--;
	}

	if(nZ)
	{
		if(!(nX0 * nX1 * nY0 * nY1))
		{
			*pNeedsZClipping = false;
			return false;
		}

		if(nZ == 8)
		{
			*pNeedsZClipping = false;
			return true;
		}

		*pNeedsZClipping = true;
		return true;
	}
	else
	{
		*pNeedsZClipping = false;
		return false;
	}
}
*/

#endif // AABB_H_