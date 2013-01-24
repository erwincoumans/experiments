/******************************************************************************

 @File         PVRTModelPODWriteH.cpp

 @Title        

 @Copyright    Copyright (C) 2003 - 2008 by Imagination Technologies Limited.

 @Platform     ANSI compatible

 @Description  Code to load POD files - models exported from MAX.

******************************************************************************/


/*
All changes:
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
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "Mathematics.h"
#include "Vertex.h"
#include "BoneBatch.h"
#include "ModelPOD.h"

#include "Macros.h"

#include <ctype.h> /*For toupper */
/****************************************************************************
** Defines
****************************************************************************/
#define CFAH		(1024)

/****************************************************************************
** Local code: File writing
****************************************************************************/

void ConvertFileToDefine(
	char		* const out,
	size_t		max,
	const char	*in)
{
	unsigned int	i;
	const char		*ptr;

	// Only take the filename portion from the end
#ifdef WIN32
	ptr = strrchr(in, '\\');
#else
	ptr = strrchr(in, '/');
#endif

	if(ptr)
	{
		++ptr;
		in = ptr;
	}

	// Cut the string down to the maximum size
	i = (unsigned int)strlen(in)+1;
	if(i > (unsigned int)max)
		i = (unsigned int)max;
	strncpy(out, in, i);
	out[--i] = 0;

	while(i)
	{
		--i;

		if(out[i] >= 'a' && out[i] <= 'z')
			out[i] = toupper(out[i]);
		else if(!(out[i] >= 'A' && out[i] <= 'Z') && !(out[i] >= '0' && out[i] <= '9'))
			out[i] = '_';
	}
}

static void WriteHData(
	FILE				* const pFile,
	const unsigned long	* const p,
	const unsigned int	nBytes,
	const char			* const sName,
	const bool bChangeEndian)
{
	unsigned int i, j;
	unsigned int n = (nBytes + 3) / 4;
	unsigned long	*pTmp;

	/*
		We're rounding up the data to some number of 'unsigned longs'. Need to
		ensure the last unnecessary 0..3 bytes bytes are 0.
	*/
	pTmp = new unsigned long[n];
	pTmp[n-1] = 0;
	memcpy(pTmp, p, nBytes);

	/*
		Write the data.
	*/
	fprintf(pFile, "const unsigned long %s[%d] =\n", sName, n);
	fprintf(pFile, "{\n");
	for(i = 0; i < n;)
	{
		fprintf(pFile, "	");
		for(j = 0; j < 128 && i < n; ++i, ++j)
		{
			fprintf(pFile, "0x%08x,", bChangeEndian ? PVRTByteSwap32(pTmp[i]) : pTmp[i]);
		}
		fprintf(pFile, "\n");
	}
	fprintf(pFile, "};\n");
	fprintf(pFile, "\n");

	/*
		Done.
	*/
	delete [] pTmp;
}

static void WriteHCameras(
	FILE			* const pFile,
	const SPODScene	&s,
	const char		* const sName,
	const bool		bChangeEndian)
{
	char			*pStr;
	unsigned int	i;

	pStr = new char[s.nNumCamera * CFAH];
	_ASSERT(pStr);

	for(i = 0; i < s.nNumCamera; ++i)
	{
		if(!s.pCamera[i].pfAnimFOV)
		{
			strcpy(&pStr[i * CFAH], "0");
		}
		else
		{
			sprintf(&pStr[i * CFAH], "%s%dAnimFOV", sName, i);
			WriteHData(pFile, (unsigned long*)s.pCamera[i].pfAnimFOV, s.nNumFrame * sizeof(*s.pCamera[i].pfAnimFOV), &pStr[i * CFAH], bChangeEndian);
		}
	}

	fprintf(pFile, "const SPODCamera %s[%d] =\n{\n", sName, s.nNumCamera);
	for(i = 0; i < s.nNumCamera; ++i)
	{
		if(s.nFlags & PVRTMODELPODSF_FIXED)
			fprintf(pFile, "	{ %d, 0x%08x, 0x%08x, 0x%08x, (int*)%s },\n", s.pCamera[i].nIdxTarget, (int&)s.pCamera[i].fFOV, (int&)s.pCamera[i].fFar, (int&)s.pCamera[i].fNear, &pStr[i * CFAH]);
		else
			fprintf(pFile, "	{ %d, %ff, %ff, %ff, (float*)%s },\n", s.pCamera[i].nIdxTarget, s.pCamera[i].fFOV, s.pCamera[i].fFar, s.pCamera[i].fNear, &pStr[i * CFAH]);
	}
	fprintf(pFile, "};\n\n");
	delete [] pStr;
}

static void WriteHLights(
	FILE			* const pFile,
	const SPODScene	&s,
	const char		* const sName)
{
	fprintf(pFile, "const SPODLight %s[%d] =\n{\n", sName, s.nNumLight);
	for(unsigned int i = 0; i < s.nNumLight; ++i)
	{
		if(s.nFlags & PVRTMODELPODSF_FIXED)
			fprintf(pFile, "	{ %d, { 0x%08x, 0x%08x, 0x%08x }, (EPODLight)%08x },\n", s.pLight[i].nIdxTarget, (int&)s.pLight[i].pfColour[0], (int&)s.pLight[i].pfColour[1], (int&)s.pLight[i].pfColour[2], s.pLight[i].eType);
		else
			fprintf(pFile, "	{ %d, { %ff, %ff, %ff }, (EPODLight)%08x },\n", s.pLight[i].nIdxTarget, s.pLight[i].pfColour[0], s.pLight[i].pfColour[1], s.pLight[i].pfColour[2], s.pLight[i].eType);
	}
	fprintf(pFile, "};\n\n");
}

static void WriteHCPODData(
	FILE				* const pFile,
	char				* const pStr,
	const CPODData		&s,
	const unsigned int	n,
	const bool			bValidData,
	const bool			bChangeEndian,
	const char			* const sName,
	const char			* const sNamePost)
{
	char buf[CFAH];

	if(bValidData && n && s.nStride) {
		sprintf(buf, "%s%s", sName, sNamePost);
		WriteHData(pFile, (unsigned long*)s.pData, n * s.nStride, buf, bChangeEndian);
		sprintf(pStr, "%s		{ (EPVRTDataType)0x%08x, %d, %d, (unsigned char*)%s },\n", pStr, s.eType, s.n, s.nStride, buf);
	} else {
		sprintf(pStr, "%s		{ (EPVRTDataType)0x%08x, %d, %d, (unsigned char*)0x%08x },\n", pStr, s.eType, s.n, s.nStride, (INT64BIT)s.pData);
	}
}

static void WriteHUVW(
	FILE			* const pFile,
	const SPODMesh	&s,
	const bool		bValidData,
	const bool		bBigEndian,
	const char		* const sName)
{
	char *pStr, buf[CFAH];

	pStr = new char[s.nNumUVW * CFAH];
	_ASSERT(pStr);

	*pStr = 0;

	for(unsigned int i = 0; i < s.nNumUVW; ++i)
	{
		sprintf(buf, "%d", i);
		WriteHCPODData(pFile, pStr, s.psUVW[i], s.nNumVertex, bValidData, bBigEndian, sName, buf);
	}

	fprintf(pFile, "const CPODData %s[%d] =\n{\n", sName, s.nNumUVW);
	fprintf(pFile, pStr);
	fprintf(pFile, "};\n\n");
	delete [] pStr;
}

static void WriteHMeshes(
	FILE			* const pFile,
	const SPODScene	&s,
	const char		* const sName,
	const bool		bChangeEndian)
{
	char *pStr, buf[CFAH];

	pStr = new char[s.nNumMesh * CFAH];
	_ASSERT(pStr);

	*pStr = 0;

	for(unsigned int i = 0; i < s.nNumMesh; ++i)
	{
		strcat(pStr, "	{\n");
		sprintf(pStr, "%s		%d,\n", pStr, s.pMesh[i].nNumVertex);
		sprintf(pStr, "%s		%d,\n", pStr, s.pMesh[i].nNumFaces);
		sprintf(pStr, "%s		%d,\n", pStr, s.pMesh[i].nNumUVW);

		sprintf(buf, "%s%d", sName, i);
		WriteHCPODData(pFile, pStr, s.pMesh[i].sFaces, PVRTModelPODCountIndices(s.pMesh[i]), true, bChangeEndian, buf, "sFaces");

		if(s.pMesh[i].nNumStrips)
		{
			sprintf(buf, "%s%dpnStripLength", sName, i);
			WriteHData(pFile, (unsigned long*)s.pMesh[i].pnStripLength, s.pMesh[i].nNumStrips * sizeof(*s.pMesh[i].pnStripLength), buf, bChangeEndian);
			sprintf(pStr, "%s		(unsigned int*)%s, %d,\n", pStr, buf, s.pMesh[i].nNumStrips);
		}
		else
		{
			sprintf(pStr, "%s		(unsigned int*)0, 0,\n", pStr);
		}

		sprintf(buf, "%s%d", sName, i);
		WriteHCPODData(pFile, pStr, s.pMesh[i].sVertex, s.pMesh[i].nNumVertex, s.pMesh[i].pInterleaved == 0, bChangeEndian, buf, "sVertex");
		WriteHCPODData(pFile, pStr, s.pMesh[i].sNormals, s.pMesh[i].nNumVertex, s.pMesh[i].pInterleaved == 0, bChangeEndian, buf, "sNormals");
		WriteHCPODData(pFile, pStr, s.pMesh[i].sTangents, s.pMesh[i].nNumVertex, s.pMesh[i].pInterleaved == 0, bChangeEndian, buf, "sTangents");
		WriteHCPODData(pFile, pStr, s.pMesh[i].sBinormals, s.pMesh[i].nNumVertex, s.pMesh[i].pInterleaved == 0, bChangeEndian, buf, "sBinormals");
		if(s.pMesh[i].nNumUVW)
		{
			sprintf(buf, "%s%dpsUVW", sName, i);
			WriteHUVW(pFile, s.pMesh[i], s.pMesh[i].pInterleaved == 0, bChangeEndian, buf);
			sprintf(pStr, "%s		(CPODData*)%s,\n", pStr, buf);
		}
		else
		{
			sprintf(pStr, "%s		(CPODData*)0,\n", pStr);
		}
		sprintf(buf, "%s%d", sName, i);
		WriteHCPODData(pFile, pStr, s.pMesh[i].sVtxColours, s.pMesh[i].nNumVertex, s.pMesh[i].pInterleaved == 0, bChangeEndian, buf, "sVtxColours");
		WriteHCPODData(pFile, pStr, s.pMesh[i].sBoneIdx, s.pMesh[i].nNumVertex, s.pMesh[i].pInterleaved == 0, bChangeEndian, buf, "sBoneIdx");
		WriteHCPODData(pFile, pStr, s.pMesh[i].sBoneWeight, s.pMesh[i].nNumVertex, s.pMesh[i].pInterleaved == 0, bChangeEndian, buf, "sBoneWeight");

		if(s.pMesh[i].pInterleaved)
		{
			sprintf(buf, "%s%dpInterleaved", sName, i);
			WriteHData(pFile, (unsigned long*)s.pMesh[i].pInterleaved, s.pMesh[i].nNumVertex * s.pMesh[i].sVertex.nStride, buf, bChangeEndian);
			sprintf(pStr, "%s		(unsigned char*)%s,\n", pStr, buf);
		}
		else
		{
			sprintf(pStr, "%s		(unsigned char*)0,\n", pStr);
		}

		if(s.pMesh[i].sBoneBatches.nBatchCnt)
		{
			sprintf(pStr, "%s		{\n", pStr);

			sprintf(buf, "%s%dpnBatches", sName, i);
			WriteHData(pFile, (unsigned long*)s.pMesh[i].sBoneBatches.pnBatches, s.pMesh[i].sBoneBatches.nBatchCnt * sizeof(*s.pMesh[i].sBoneBatches.pnBatches) * s.pMesh[i].sBoneBatches.nBatchBoneMax, buf, bChangeEndian);
			sprintf(pStr, "%s			(int*)%s,\n", pStr, buf);

			sprintf(buf, "%s%dpnBatchBoneCnt", sName, i);
			WriteHData(pFile, (unsigned long*)s.pMesh[i].sBoneBatches.pnBatchBoneCnt, s.pMesh[i].sBoneBatches.nBatchCnt * sizeof(*s.pMesh[i].sBoneBatches.pnBatchBoneCnt), buf, bChangeEndian);
			sprintf(pStr, "%s			(int*)%s,\n", pStr, buf);

			sprintf(buf, "%s%dpnBatchOffset", sName, i);
			WriteHData(pFile, (unsigned long*)s.pMesh[i].sBoneBatches.pnBatchOffset, s.pMesh[i].sBoneBatches.nBatchCnt * sizeof(*s.pMesh[i].sBoneBatches.pnBatchOffset), buf, bChangeEndian);
			sprintf(pStr, "%s			(int*)%s,\n", pStr, buf);

			sprintf(pStr, "%s			%d,\n", pStr, s.pMesh[i].sBoneBatches.nBatchBoneMax);
			sprintf(pStr, "%s			%d,\n", pStr, s.pMesh[i].sBoneBatches.nBatchCnt);
			sprintf(pStr, "%s		}\n", pStr);
		}
		else
		{
			sprintf(pStr, "%s		{ (int*)0, (int*)0, (int*)0, 0, 0 },\n", pStr);
		}

		strcat(pStr, "	},\n");
	}

	fprintf(pFile, "const SPODMesh %s[%d] =\n{\n", sName, s.nNumMesh);
	fprintf(pFile, pStr);
	fprintf(pFile, "};\n\n");
	delete [] pStr;
}

static void WriteHNodes(
	FILE			* const pFile,
	const SPODScene	&s,
	const char		* const sName,
	const bool bChangeEndian)
{
	char *pStr, buf[CFAH];
	int i32TransformNo;

	pStr = new char[s.nNumNode * CFAH];
	_ASSERT(pStr);

	*pStr = 0;

	for(unsigned int i = 0; i < s.nNumNode; ++i)
	{
		strcat(pStr, "	{\n");
		sprintf(pStr, "%s		%d,\n", pStr, s.pNode[i].nIdx);

		strcpy(buf, "0");
		if(s.pNode[i].pszName)
		{
			sprintf(buf, "%s%dpszName", sName, i);
			WriteHData(pFile, (unsigned long*)s.pNode[i].pszName, (unsigned int)strlen(s.pNode[i].pszName) + 1, buf, bChangeEndian);
		}
		sprintf(pStr, "%s		(char*)%s, /* %s */\n", pStr, buf, s.pNode[i].pszName ? s.pNode[i].pszName : "");

		sprintf(pStr, "%s		%d,\n", pStr, s.pNode[i].nIdxMaterial);
		sprintf(pStr, "%s		%d,\n", pStr, s.pNode[i].nIdxParent);
		sprintf(pStr, "%s		%d,\n", pStr, s.pNode[i].nAnimFlags);

		strcpy(buf, "0");
		if(s.pNode[i].pfAnimPosition)
		{
			if(s.pNode[i].nAnimFlags & ePODHasPositionAni)
			{
				sprintf(buf, "%s%dpfAnimPosition", sName, i);
				i32TransformNo = s.nNumFrame;
		}
		else
		{
				sprintf(buf, "%s%dpfPosition", sName, i);
				i32TransformNo = 1;
		}

			WriteHData(pFile, (unsigned long*)s.pNode[i].pfAnimPosition, i32TransformNo * 3 * sizeof(*s.pNode[i].pfAnimPosition), buf, bChangeEndian);
		}
		if(s.nFlags & PVRTMODELPODSF_FIXED)
			sprintf(pStr, "%s		(int*)%s,\n", pStr, buf);
		else
			sprintf(pStr, "%s		(float*)%s,\n", pStr, buf);

		strcpy(buf, "0");
		if(s.pNode[i].pfAnimRotation)
		{
			if(s.pNode[i].nAnimFlags & ePODHasRotationAni)
			{
			sprintf(buf, "%s%dpfAnimRotation", sName, i);
				i32TransformNo = s.nNumFrame;
		}
			else
			{
				sprintf(buf, "%s%dpfRotation", sName, i);
				i32TransformNo = 1;
			}

			WriteHData(pFile, (unsigned long*)s.pNode[i].pfAnimRotation, i32TransformNo * 4 * sizeof(*s.pNode[i].pfAnimRotation), buf, bChangeEndian);
		}
		if(s.nFlags & PVRTMODELPODSF_FIXED)
			sprintf(pStr, "%s		(int*)%s,\n", pStr, buf);
		else
			sprintf(pStr, "%s		(float*)%s,\n", pStr, buf);

		strcpy(buf, "0");
		if(s.pNode[i].pfAnimScale)
		{
			if(s.pNode[i].nAnimFlags & ePODHasScaleAni)
			{
			sprintf(buf, "%s%dpfAnimScale", sName, i);
				i32TransformNo = s.nNumFrame;
			}
			else
			{
				sprintf(buf, "%s%dpfScale", sName, i);
				i32TransformNo = 1;
		}

			WriteHData(pFile, (unsigned long*)s.pNode[i].pfAnimScale, i32TransformNo * 7 * sizeof(*s.pNode[i].pfAnimScale), buf, bChangeEndian);
		}
		if(s.nFlags & PVRTMODELPODSF_FIXED)
			sprintf(pStr, "%s		(int*)%s,\n", pStr, buf);
		else
			sprintf(pStr, "%s		(float*)%s,\n", pStr, buf);

		strcpy(buf, "0");
		if(s.pNode[i].pfAnimMatrix)
		{
			if(s.pNode[i].nAnimFlags & ePODHasMatrixAni)
			{
				sprintf(buf, "%s%dpfAnimMatrix", sName, i);
				i32TransformNo = s.nNumFrame;
			}
			else
			{
				sprintf(buf, "%s%dpfMatrix", sName, i);
				i32TransformNo = 1;
			}

			WriteHData(pFile, (unsigned long*)s.pNode[i].pfAnimMatrix, i32TransformNo * 16 * sizeof(*s.pNode[i].pfAnimMatrix), buf, bChangeEndian);
		}
		if(s.nFlags & PVRTMODELPODSF_FIXED)
			sprintf(pStr, "%s		(int*)%s,\n", pStr, buf);
		else
			sprintf(pStr, "%s		(float*)%s,\n", pStr, buf);

		strcat(pStr, "	},\n");
	}

	fprintf(pFile, "const SPODNode %s[%d] =\n{\n", sName, s.nNumNode);
	fprintf(pFile, pStr);
	fprintf(pFile, "};\n\n");
	delete [] pStr;
}

static void WriteHTextures(
	FILE			* const pFile,
	const SPODScene	&s,
	const char		* const sName,
	const bool		bChangeEndian)
{
	char *pStr, buf[CFAH];

	pStr = new char[s.nNumTexture * CFAH];
	_ASSERT(pStr);

	*pStr = 0;

	for(unsigned int i = 0; i < s.nNumTexture; ++i)
	{
		strcat(pStr, "	{\n");

		strcpy(buf, "0");
		if(s.pTexture[i].pszName)
		{
			sprintf(buf, "%s%dpszName", sName, i);
			WriteHData(pFile, (unsigned long*)s.pTexture[i].pszName, (unsigned int)strlen(s.pTexture[i].pszName) + 1, buf, bChangeEndian);
		}
		sprintf(pStr, "%s		(char*)%s,\n", pStr, buf);

		strcat(pStr, "	},\n");
	}

	fprintf(pFile, "const SPODTexture %s[%d] =\n{\n", sName, s.nNumTexture);
	fprintf(pFile, pStr);
	fprintf(pFile, "};\n\n");
	delete [] pStr;
}

static void WriteHMaterials(
	FILE			* const pFile,
	const SPODScene	&s,
	const char		* const sName,
	const bool		bChangeEndian)
{
	char *pStr, buf[CFAH];

	pStr = new char[s.nNumMaterial * CFAH];
	_ASSERT(pStr);

	*pStr = 0;

	for(unsigned int i = 0; i < s.nNumMaterial; ++i)
	{
		strcat(pStr, "	{\n");

		strcpy(buf, "0");
		if(s.pMaterial[i].pszName)
		{
			sprintf(buf, "%s%dpszName", sName, i);
			WriteHData(pFile, (unsigned long*)s.pMaterial[i].pszName, (unsigned int)strlen(s.pMaterial[i].pszName) + 1, buf, bChangeEndian);
		}
		sprintf(pStr, "%s		(char*)%s, /* %s */\n", pStr, buf, s.pMaterial[i].pszName ? s.pMaterial[i].pszName : "");

		sprintf(pStr, "%s		%d,\n", pStr, s.pMaterial[i].nIdxTexDiffuse);
		if(s.nFlags & PVRTMODELPODSF_FIXED)
		{
			sprintf(pStr, "%s		0x%08x,\n", pStr, (int&)s.pMaterial[i].fMatOpacity);
			sprintf(pStr, "%s		{ 0x%08x, 0x%08x, 0x%08x },\n", pStr, (int&)s.pMaterial[i].pfMatAmbient[0], (int&)s.pMaterial[i].pfMatAmbient[1], (int&)s.pMaterial[i].pfMatAmbient[2]);
			sprintf(pStr, "%s		{ 0x%08x, 0x%08x, 0x%08x },\n", pStr, (int&)s.pMaterial[i].pfMatDiffuse[0], (int&)s.pMaterial[i].pfMatDiffuse[1], (int&)s.pMaterial[i].pfMatDiffuse[2]);
			sprintf(pStr, "%s		{ 0x%08x, 0x%08x, 0x%08x },\n", pStr, (int&)s.pMaterial[i].pfMatSpecular[0], (int&)s.pMaterial[i].pfMatSpecular[1], (int&)s.pMaterial[i].pfMatSpecular[2]);
			sprintf(pStr, "%s		0x%08x,\n", pStr, (int&)s.pMaterial[i].fMatShininess);
		}
		else
		{
			sprintf(pStr, "%s		%ff,\n", pStr, s.pMaterial[i].fMatOpacity);
			sprintf(pStr, "%s		{ %ff, %ff, %ff },\n", pStr, s.pMaterial[i].pfMatAmbient[0], s.pMaterial[i].pfMatAmbient[1], s.pMaterial[i].pfMatAmbient[2]);
			sprintf(pStr, "%s		{ %ff, %ff, %ff },\n", pStr, s.pMaterial[i].pfMatDiffuse[0], s.pMaterial[i].pfMatDiffuse[1], s.pMaterial[i].pfMatDiffuse[2]);
			sprintf(pStr, "%s		{ %ff, %ff, %ff },\n", pStr, s.pMaterial[i].pfMatSpecular[0], s.pMaterial[i].pfMatSpecular[1], s.pMaterial[i].pfMatSpecular[2]);
			sprintf(pStr, "%s		%ff,\n", pStr, s.pMaterial[i].fMatShininess);
		}

		strcpy(buf, "0");
		if(s.pMaterial[i].pszEffectFile)
		{
			sprintf(buf, "%s%dpszEffectFile", sName, i);
			WriteHData(pFile, (unsigned long*)s.pMaterial[i].pszEffectFile, (unsigned int)strlen(s.pMaterial[i].pszEffectFile) + 1, buf, bChangeEndian);
		}
		sprintf(pStr, "%s		(char*)%s,\n", pStr, buf);

		strcpy(buf, "0");
		if(s.pMaterial[i].pszEffectName)
		{
			sprintf(buf, "%s%dpszEffectName", sName, i);
			WriteHData(pFile, (unsigned long*)s.pMaterial[i].pszEffectName, (unsigned int)strlen(s.pMaterial[i].pszEffectName) + 1, buf, bChangeEndian);
		}
		sprintf(pStr, "%s		(char*)%s,\n", pStr, buf);

		strcat(pStr, "	},\n");
	}

	fprintf(pFile, "const SPODMaterial %s[%d] =\n{\n", sName, s.nNumMaterial);
	fprintf(pFile, pStr);
	fprintf(pFile, "};\n\n");
	delete [] pStr;
}

static void WriteHScene(
	FILE			* const pFile,
	const SPODScene	&s,
	const char		* const sName)
{
	char sCamera[CFAH] = "0";
	char sLight[CFAH] = "0";
	char sMesh[CFAH] = "0";
	char sNode[CFAH] = "0";
	char sTexture[CFAH] = "0";
	char sMaterial[CFAH] = "0";

	bool bChangeEndian = s.bBigEndian == PVRTIsLittleEndian();

	if(s.nNumCamera)
	{
		sprintf(sCamera, "%sCamera", sName);
		WriteHCameras(pFile, s, sCamera, bChangeEndian);
	}

	if(s.nNumLight)
	{
		sprintf(sLight, "%sLight", sName);
		WriteHLights(pFile, s, sLight);
	}

	if(s.nNumMesh)
	{
		sprintf(sMesh, "%sMesh", sName);
		WriteHMeshes(pFile, s, sMesh, bChangeEndian);
	}

	if(s.nNumNode)
	{
		sprintf(sNode, "%sNode", sName);
		WriteHNodes(pFile, s, sNode, bChangeEndian);
	}

	if(s.nNumTexture)
	{
		sprintf(sTexture, "%sTexture", sName);
		WriteHTextures(pFile, s, sTexture, bChangeEndian);
	}

	if(s.nNumMaterial)
	{
		sprintf(sMaterial, "%sMaterial", sName);
		WriteHMaterials(pFile, s, sMaterial, bChangeEndian);
	}

	fprintf(pFile, "const SPODScene %s =\n", sName);
	fprintf(pFile, "{\n");
	if(s.nFlags & PVRTMODELPODSF_FIXED)
	{
		fprintf(pFile, "	{ 0x%08x, 0x%08x, 0x%08x },\n", (int&)s.pfColourBackground[0], (int&)s.pfColourBackground[1], (int&)s.pfColourBackground[2]);
		fprintf(pFile, "	{ 0x%08x, 0x%08x, 0x%08x },\n", (int&)s.pfColourAmbient[0], (int&)s.pfColourAmbient[1], (int&)s.pfColourAmbient[2]);
	}
	else
	{
		fprintf(pFile, "	{ %ff, %ff, %ff },\n", s.pfColourBackground[0], s.pfColourBackground[1], s.pfColourBackground[2]);
		fprintf(pFile, "	{ %ff, %ff, %ff },\n", s.pfColourAmbient[0], s.pfColourAmbient[1], s.pfColourAmbient[2]);
	}
	fprintf(pFile, "	%d,\n", s.nNumCamera);
	fprintf(pFile, "	(SPODCamera*)%s,\n", sCamera);
	fprintf(pFile, "	%d,\n", s.nNumLight);
	fprintf(pFile, "	(SPODLight*)%s,\n", sLight);
	fprintf(pFile, "	%d,\n", s.nNumMesh);
	fprintf(pFile, "	(SPODMesh*)%s,\n", sMesh);
	fprintf(pFile, "	%d,\n", s.nNumNode);
	fprintf(pFile, "	%d,\n", s.nNumMeshNode);
	fprintf(pFile, "	(SPODNode*)%s,\n", sNode);
	fprintf(pFile, "	%d,\n", s.nNumTexture);
	fprintf(pFile, "	(SPODTexture*)%s,\n", sTexture);
	fprintf(pFile, "	%d,\n", s.nNumMaterial);
	fprintf(pFile, "	(SPODMaterial*)%s,\n", sMaterial);
	fprintf(pFile, "	%d,\n", s.nNumFrame);
	fprintf(pFile, "	0x%08x,\n", s.nFlags);
	fprintf(pFile, "};\n");
	fprintf(pFile, "\n");
}

/*!***************************************************************************
 @Function			WriteH
 @Output			pFile
 @Input				pszFilename
 @Input				pszExpOpt
 @Input				s
 @Return			true if successful
 @Description		Write a POD file
*****************************************************************************/
static bool WriteH(
	FILE			* const pFile,
	const char		* const pszFilename,
	const char		* const pszExpOpt,
	const SPODScene	&s)
{
	time_t		expTime = time(NULL);
	char		buf[CFAH], tmp[CFAH];
	const char	*ptr;

	ConvertFileToDefine(tmp, CFAH, pszFilename);
	ptr = strrchr(pszFilename, '\\');
	if(ptr)
		++ptr;
	else
		ptr = pszFilename;

	fprintf(pFile, "/*\n");
	fprintf(pFile, "	POD header-file, as from PVRTModelPOD and thereby PVRGeoPOD.\n");
	fprintf(pFile, "	File-format version string: \"%s\"\n", PVRTMODELPOD_VERSION);
	fprintf(pFile, "	This code was compiled: %s %s\n", __TIME__, __DATE__);
	fprintf(pFile, "\n");
	fprintf(pFile, "	Export name: \"%s\"\n", ptr);
	fprintf(pFile, "	Export time: %s", ctime(&expTime));
	fprintf(pFile, "\n");
	fprintf(pFile, "	ExpOpt{\n");
	fprintf(pFile, pszExpOpt ? pszExpOpt : "");
	fprintf(pFile, "	}ExpOpt\n");
	fprintf(pFile, "*/\n");
	fprintf(pFile, "\n");
	fprintf(pFile, "#ifndef _%s_\n", tmp);
	fprintf(pFile, "#define _%s_\n", tmp);
	fprintf(pFile, "\n");
	fprintf(pFile, "\n");
	fprintf(pFile, "#if !defined(_PVRTMODELPOD_H_)\n");
	fprintf(pFile, "#error Before including this file, you must include a header to define the POD structures.\n");
	fprintf(pFile, "#endif\n");
	fprintf(pFile, "\n");

	sprintf(buf, "c_%s", tmp);
	WriteHScene(pFile, s, buf);

	fprintf(pFile, "\n");
	fprintf(pFile, "#endif /* (_%s_) */\n", tmp);
	fprintf(pFile, "\n");
	fprintf(pFile, "/*\n");
	fprintf(pFile, "	End of file: %s\n", ptr);
	fprintf(pFile, "*/\n");
	return true;
}

/****************************************************************************
** Class: CPVRTModelPOD
****************************************************************************/

/*!***********************************************************************
 @Function			SaveH
 @Input				pszFilename		Filename to save to
 @Description		Save a header file (.H).
*************************************************************************/
bool CPVRTModelPOD::SaveH(const char * const pszFilename, const char * const pszExpOpt)
{
	FILE	*pFile;
	bool	bRet;

	pFile = fopen(pszFilename, "wt+");
	if(!pFile)
		return false;

	bRet = WriteH(pFile, pszFilename, pszExpOpt, *this);

	// Done
	fclose(pFile);
	return bRet ? true : false;
}

/****************************************************************************
** Code
****************************************************************************/

/*****************************************************************************
 End of file (PVRTModelPODWriteH.cpp)
*****************************************************************************/
