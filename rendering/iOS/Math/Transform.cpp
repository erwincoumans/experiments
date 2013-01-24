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

#include <string.h>

//#include "FixedPoint.h"
#include "Matrix.h"
#include "Transform.h"


/*!***************************************************************************
 @Function Name		TransVec3TransformArray
 @Output			pOut				Destination for transformed vectors
 @Input				nOutStride			Stride between vectors in pOut array
 @Input				pV					Input vector array
 @Input				nInStride			Stride between vectors in pV array
 @Input				pMatrix				Matrix to transform the vectors
 @Input				nNumberOfVertices	Number of vectors to transform
 @Description		Transform all vertices [X Y Z 1] in pV by pMatrix and
 					store them in pOut.
*****************************************************************************/
void TransVec3TransformArray(
	VECTOR4			* const pOut,
	const int			nOutStride,
	const VECTOR3	* const pV,
	const int			nInStride,
	const MATRIX	* const pMatrix,
	const int			nNumberOfVertices)
{
	const VECTOR3	*pSrc;
	VECTOR4			*pDst;
	int					i;

	pSrc = pV;
	pDst = pOut;

	/* Transform all vertices with *pMatrix */
	for (i=0; i<nNumberOfVertices; ++i)
	{
		pDst->x =	VERTTYPEMUL(pMatrix->f[ 0], pSrc->x) +
					VERTTYPEMUL(pMatrix->f[ 4], pSrc->y) +
					VERTTYPEMUL(pMatrix->f[ 8], pSrc->z) +
					pMatrix->f[12];
		pDst->y =	VERTTYPEMUL(pMatrix->f[ 1], pSrc->x) +
					VERTTYPEMUL(pMatrix->f[ 5], pSrc->y) +
					VERTTYPEMUL(pMatrix->f[ 9], pSrc->z) +
					pMatrix->f[13];
		pDst->z =	VERTTYPEMUL(pMatrix->f[ 2], pSrc->x) +
					VERTTYPEMUL(pMatrix->f[ 6], pSrc->y) +
					VERTTYPEMUL(pMatrix->f[10], pSrc->z) +
					pMatrix->f[14];
		pDst->w =	VERTTYPEMUL(pMatrix->f[ 3], pSrc->x) +
					VERTTYPEMUL(pMatrix->f[ 7], pSrc->y) +
					VERTTYPEMUL(pMatrix->f[11], pSrc->z) +
					pMatrix->f[15];

		pDst = (VECTOR4*)((char*)pDst + nOutStride);
		pSrc = (VECTOR3*)((char*)pSrc + nInStride);
	}
}

/*!***************************************************************************
 @Function			TransTransformArray
 @Output			pTransformedVertex	Destination for transformed vectors
 @Input				pV					Input vector array
 @Input				nNumberOfVertices	Number of vectors to transform
 @Input				pMatrix				Matrix to transform the vectors
 @Input				fW					W coordinate of input vector (e.g. use 1 for position, 0 for normal)
 @Description		Transform all vertices in pVertex by pMatrix and store them in
					pTransformedVertex
					- pTransformedVertex is the pointer that will receive transformed vertices.
					- pVertex is the pointer to untransformed object vertices.
					- nNumberOfVertices is the number of vertices of the object.
					- pMatrix is the matrix used to transform the object.
*****************************************************************************/
void TransTransformArray(
	VECTOR3			* const pTransformedVertex,
	const VECTOR3	* const pV,
	const int			nNumberOfVertices,
	const MATRIX	* const pMatrix,
	const VERTTYPE		fW)
{
	int			i;

	/* Transform all vertices with *pMatrix */
	for (i=0; i<nNumberOfVertices; ++i)
	{
		pTransformedVertex[i].x =	VERTTYPEMUL(pMatrix->f[ 0], pV[i].x) +
									VERTTYPEMUL(pMatrix->f[ 4], pV[i].y) +
									VERTTYPEMUL(pMatrix->f[ 8], pV[i].z) +
									VERTTYPEMUL(pMatrix->f[12], fW);
		pTransformedVertex[i].y =	VERTTYPEMUL(pMatrix->f[ 1], pV[i].x) +
									VERTTYPEMUL(pMatrix->f[ 5], pV[i].y) +
									VERTTYPEMUL(pMatrix->f[ 9], pV[i].z) +
									VERTTYPEMUL(pMatrix->f[13], fW);
		pTransformedVertex[i].z =	VERTTYPEMUL(pMatrix->f[ 2], pV[i].x) +
									VERTTYPEMUL(pMatrix->f[ 6], pV[i].y) +
									VERTTYPEMUL(pMatrix->f[10], pV[i].z) +
									VERTTYPEMUL(pMatrix->f[14], fW);
	}
}

/*!***************************************************************************
 @Function			TransTransformArrayBack
 @Output			pTransformedVertex
 @Input				pVertex
 @Input				nNumberOfVertices
 @Input				pMatrix
 @Description		Transform all vertices in pVertex by the inverse of pMatrix
					and store them in pTransformedVertex.
					- pTransformedVertex is the pointer that will receive transformed vertices.
					- pVertex is the pointer to untransformed object vertices.
					- nNumberOfVertices is the number of vertices of the object.
					- pMatrix is the matrix used to transform the object.
*****************************************************************************/
void TransTransformArrayBack(
	VECTOR3			* const pTransformedVertex,
	const VECTOR3	* const pVertex,
	const int			nNumberOfVertices,
	const MATRIX	* const pMatrix)
{
	MATRIX	mBack;

	MatrixInverse(mBack, *pMatrix);
	TransTransformArray(pTransformedVertex, pVertex, nNumberOfVertices, &mBack);
}

/*!***************************************************************************
 @Function			TransTransformBack
 @Output			pOut
 @Input				pV
 @Input				pM
 @Description		Transform vertex pV by the inverse of pMatrix
					and store in pOut.
*****************************************************************************/
void TransTransformBack(
	VECTOR4			* const pOut,
	const VECTOR4	* const pV,
	const MATRIX	* const pM)
{
	VERTTYPE *ppfRows[4];
	VERTTYPE pfIn[20];
	int i;
	const MATRIX	*pMa;

#if defined(BUILD_OGL) || defined(BUILD_OGLES) || defined(BUILD_OGLES2)
	MATRIX mT;
	MatrixTranspose(mT, *pM);
	pMa = &mT;
#else
	pMa = pM;
#endif

	for(i = 0; i < 4; ++i)
	{
		/*
			Set up the array of pointers to matrix coefficients
		*/
		ppfRows[i] = &pfIn[i * 5];

		/*
			Copy the 4x4 matrix into RHS of the 5x4 matrix
		*/
		memcpy(&ppfRows[i][1], &pMa->f[i * 4], 4 * sizeof(float));
	}

	/*
		Copy the "result" vector into the first column of the 5x4 matrix
	*/
	ppfRows[0][0] = pV->x;
	ppfRows[1][0] = pV->y;
	ppfRows[2][0] = pV->z;
	ppfRows[3][0] = pV->w;

	/*
		Solve a set of 4 linear equations
	*/
	MatrixLinearEqSolve(&pOut->x, ppfRows, 4);
}

/*!***************************************************************************
 @Function			TransTransform
 @Output			pOut
 @Input				pV
 @Input				pM
 @Description		Transform vertex pV by pMatrix and store in pOut.
*****************************************************************************/
void TransTransform(
	VECTOR4			* const pOut,
	const VECTOR4	* const pV,
	const MATRIX	* const pM)
{
	pOut->x = pM->f[0] * pV->x + pM->f[4] * pV->y + pM->f[8] * pV->z + pM->f[12] * pV->w;
	pOut->y = pM->f[1] * pV->x + pM->f[5] * pV->y + pM->f[9] * pV->z + pM->f[13] * pV->w;
	pOut->z = pM->f[2] * pV->x + pM->f[6] * pV->y + pM->f[10] * pV->z + pM->f[14] * pV->w;
	pOut->w = pM->f[3] * pV->x + pM->f[7] * pV->y + pM->f[11] * pV->z + pM->f[15] * pV->w;
}
