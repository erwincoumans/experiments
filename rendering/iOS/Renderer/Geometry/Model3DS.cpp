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

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "Mathematics.h"
#include "Geometry.h"

#include "MemoryManager.h"
#include "Macros.h"

#include "Model3DS.h"

/***********
** Macros **
***********/
//#define PI		3.14159f


/************
** Defines **
************/
#define DIAGNOSTICS				false
#define MAX_MESHES				4096
#define MAX_NODES				300

#define M3DMAGIC				0x4D4D
#define M3D_VERSION				0x0002
#define MDATA					0x3D3D
#define MASTER_SCALE			0x0100
#define KFDATA					0xB000
#define NAMED_OBJECT			0x4000
#define N_TRI_OBJECT			0x4100
#define	N_DIRECT_LIGHT			0x4600
#define N_CAMERA				0x4700
#define POINT_ARRAY				0x4110
#define FACE_ARRAY				0x4120
#define TEX_VERTS				0x4140
#define MESH_MATRIX				0x4160

#define MSH_MAT_GROUP			0x4130
#define MAT_ENTRY				0xAFFF
#define MAT_NAME				0xA000
#define MAT_AMBIENT				0xA010
#define MAT_DIFFUSE				0xA020
#define MAT_SPECULAR			0xA030
#define MAT_SHININESS			0xA040
#define MAT_TRANSPARENCY		0xA050
#define MAT_SHADING				0xA100
#define MAT_TEXMAP				0xA200
#define MAT_OPACMAP				0xA210

#define KFHDR					0xB00A
#define KFSEG					0xB008
#define KFCURTIME				0xB009
#define OBJECT_NODE_TAG			0xB002
#define CAMERA_NODE_TAG			0xB003
#define TARGET_NODE_TAG			0xB004
#define LIGHT_NODE_TAG			0xB005
#define SPOTLIGHT_NODE_TAG		0xB007
#define L_TARGET_NODE_TAG		0xB006
#define NODE_HDR				0xB010
#define PIVOT					0xB013
#define INSTANCE_NAME			0xB011
#define MORPH_SMOOTH			0xB015
#define BOUNDBOX				0xB014
#define POS_TRACK_TAG			0xB020
#define COL_TRACK_TAG			0xB025
#define ROT_TRACK_TAG			0xB021
#define SCL_TRACK_TAG			0xB022
#define MORPH_TRACK_TAG			0xB026
#define FOV_TRACK_TAG			0xB023
#define ROLL_TRACK_TAG			0xB024
#define HOT_TRACK_TAG			0xB027
#define FALL_TRACK_TAG			0xB028
#define NODE_ID					0xB030

#define INT_PERCENTAGE			0x0030
#define FLOAT_PERCENTAGE		0x0031
#define COLOR_24				0x0011
#define COLOR_F					0x0010


//DEFINE_HEAP(C3DSScene, "Model 3DS");

/**************
** Functions **
**************/

/* For some weird reason global optimations have to be turned OFF for that function */
//#pragma optimize( "g", off )

/*!***************************************************************************
 @Function			ReadFromFile
 @Input				pszFileName			Name of the 3DS file to load
 @Return			true or false
 @Description		Read a 3DS file. The 3DS file can either be a resource or file.
					If stored in a resource, the resource identifier must be "TDS".
*****************************************************************************/
bool C3DSScene::ReadFromFile(const char * const pszFileName)
{
#define BREAD(Destination, Pointer, Size)	{ memcpy(Destination, Pointer, Size); Pointer+=Size; }
	FILE			*Fich;
#ifdef _WIN32
	HRSRC			hsSrc=NULL;
#endif
	unsigned char	*p3DSFile=NULL, *pByte;
	MATRIX		g_Matrix[MAX_MESHES], InversionMatrix;
	VECTOR3		*pSource, D;
	int				MeshPos[MAX_MESHES];
	int				MatPos[MAX_MESHES];
	int				LightPos[MAX_MESHES];
	int				CameraPos[MAX_MESHES];
	char			MeshName[MAX_MESHES][20];
	char			CameraName[MAX_MESHES][20];
	bool			bLoadFromResource=false;
	int				nMeshNumber, nMatNumber, nLightNumber, nCameraNumber;
	unsigned int	FilePos, FileSize, ChunkSize;
	int				MDATAFileOffset=-1, KFDATAFileOffset=-1;
	int				i, j;
	unsigned short	Chunk=0, TempChunk=0;
	unsigned int	Size, TempSize, SecCont=0, TmpInt;
	short			TmpShort =0;
	VERTTYPE		fMasterScale= f2vt(1.0f);
	VERTTYPE		fX, fY, fZ;
	unsigned char	TempByte[3];
	float			TempFloat[3];
	char			pszTmp[512];
	bool			bNodeTarget = 0;

	fFrame = f2vt(0.0f);
	/***************************************************
	** Copy 3DS file or resource into a memory buffer **
	***************************************************/

#if defined(_WIN32) && !defined(UNDER_CE)
	/* First determine whether the file is in resource or not */
	hsSrc = FindResourceA(GetModuleHandle(NULL), pszFileName, "TDS");
	if (hsSrc)
	{
		/* Resource case */

		/* Set boolean */
		bLoadFromResource=true;

		/* Load 3DS resource and get a pointer from it */
		p3DSFile=(unsigned char *)LoadResource(GetModuleHandle(NULL), hsSrc);
	}
	else
#endif
	{
		/* Disk case */

		/* Open 3DS file for reading in binary mode */
		if ((Fich = fopen(pszFileName, "rb")) == NULL)
		{
			sprintf(pszTmp, "Model3DSRead : Can't find %s\n", pszFileName);
			_RPT0(_CRT_WARN, pszTmp);
			return false;
		}

		/* Read file ID and size */
		fread(&Chunk,	 2, 1, Fich);
		fread(&FileSize, 4, 1, Fich);

		/* Allocate memory to store the entire file in memory */
		//p3DSFile=(unsigned char *)malloc(FileSize*sizeof(unsigned char));
		p3DSFile=new unsigned char [FileSize*sizeof(unsigned char)];

		if (!p3DSFile)
		{
			_RPT0(_CRT_WARN, "Model3DSRead : Not enough memory to store 3DS model\n");
			fclose(Fich);
			return false;
		}

		/* Reset file pointer to beginning */
		fseek(Fich, 0, SEEK_SET);

		/* Copy entire file into memory buffer */
		if (fread(p3DSFile, FileSize*sizeof(unsigned char), 1, Fich)!=1)
		{
			sprintf(pszTmp, "Model3DSRead : Error when loading file %s\n", pszFileName);
			_RPT0(_CRT_WARN, pszTmp);
			delete(p3DSFile);
			fclose(Fich);
			return false;
		}

		/* Close file */
		fclose(Fich);
	}


	/***********************************************************
	** At this point the entire 3DS file is located in memory **
	***********************************************************/

	/* Set pointer to 3DS buffer */
	pByte = p3DSFile;

	/* Read file ID and size */
	BREAD(&Chunk, pByte, sizeof(unsigned short));
	BREAD(&FileSize, pByte, sizeof(unsigned int));

	/* Check that file is a valid 3DS file */
	if(Chunk != M3DMAGIC)
	{
		_RPT0(_CRT_WARN, "Model3DSRead : This file is not a valid 3D Studio File.\n");

		if (!bLoadFromResource)
			delete(p3DSFile);

		return false;
	}


	/* File offset is 6 (just after 3DS file format ID chunk : M3DMAGIC) */
	FilePos=6;

	/* Looking for the three main chunks (M3D_VERSION, MDATA and KFDATA) */
    while(FilePos < FileSize && ++SecCont < 200)
	{
		/* Go to next chunk */
		pByte = p3DSFile + FilePos;

		/* Read chunk and size */
		BREAD(&Chunk, pByte, sizeof(unsigned short));
		BREAD(&Size,  pByte, sizeof(unsigned int));

		/* Get offsets of MDATA and KFDATA */
		switch(Chunk)
		{
			case MDATA :
				MDATAFileOffset = FilePos;
				break;
			case KFDATA :
				KFDATAFileOffset = FilePos;
				break;
			case M3D_VERSION :
				BREAD(&TmpInt, pByte, sizeof(unsigned int));
				if(DIAGNOSTICS)
				{
					sprintf(pszTmp, "M3D_VERSION = %X\n", TmpInt);
					_RPT0(_CRT_WARN,  pszTmp);
				}
				break;
			default:
				if (DIAGNOSTICS)
				{
					_RPT0(_CRT_WARN,  "Model3DSRead : Unknown M3DMAGIC chunk encountered.\n");
				}
		}

		/* Have we found what we're looking for ? */
		if (MDATAFileOffset != -1 && KFDATAFileOffset != -1)
			break;

		/* Compute next chunk position */
        FilePos += Size;
    }

	/* Check that MDATA chunk was found */
	if (MDATAFileOffset==-1)
	{
		_RPT0(_CRT_WARN,  "Model3DSRead : ERROR! MDATA chunk not found.\n");

		if (!bLoadFromResource)
			delete(p3DSFile);

		return false;
	}

	/* Did we find KFDATA ? */
	if (KFDATAFileOffset==-1)
	{
		_RPT0(_CRT_WARN,  "Model3DSRead : Model does not contain animation.\n");
	}

	/***************
	****************
	** Read MDATA **
	****************
	***************/

	/* Set pointer to start of data */
	SecCont = 0;
	FilePos = MDATAFileOffset + 6;
	pByte = p3DSFile + FilePos;

	/******************************************************************
	** Initialise chunks counters (mesh, material, light and camera) **
	******************************************************************/
	nMeshNumber =	0;
	nMatNumber =	0;
	nLightNumber =	0;
	nCameraNumber =	0;

	/*  Looking for different Meshes or Material definitions */
    while (SecCont++ < MAX_MESHES)
	{
		/* Read chunk and size */
		BREAD(&Chunk, pByte, sizeof(unsigned short));
		BREAD(&Size,  pByte, sizeof(unsigned int));

		/* Is chunk a valid one? */
		switch(Chunk)
		{
		case MASTER_SCALE:		/* Read master scale (not used) */
								BREAD(&TempFloat[0], pByte, sizeof(float));
								fMasterScale = f2vt(TempFloat[0]);
								if (DIAGNOSTICS)
								{
									sprintf(pszTmp, "Master scale is %.2f\n", fMasterScale);
									_RPT0(_CRT_WARN,  pszTmp);
								}
								break;

		case NAMED_OBJECT:		/* Check whether we've reached the maximum number of meshes */
								if (nMeshNumber >= MAX_MESHES)
								{
									_RPT0(_CRT_WARN,  "MAX_MESHES reached\n.");
									break;
								}

								/* Read mesh name */
								for (j=0; j<20; j++)
								{
									BREAD(&MeshName[nMeshNumber][j], pByte, sizeof(char));

									if (MeshName[nMeshNumber][j]==0)
										break;

								}

								/* Read chunk */
								BREAD(&Chunk, pByte, sizeof(unsigned short));

								switch(Chunk)
								{
								case N_TRI_OBJECT:		/* Object geometry */
														MeshPos[nMeshNumber] = FilePos + 7 + j;
														++nMeshNumber;
														break;

								case N_DIRECT_LIGHT:	/* Object light */
														LightPos[nLightNumber] = FilePos + 7 + j;
														++nLightNumber;
														break;

								case N_CAMERA:			/* Object camera */
														// copy the mesh name into the especific field in the camera struct
														strcpy(CameraName[nCameraNumber], MeshName[nMeshNumber]);

														CameraPos[nCameraNumber] = FilePos + 7 + j;
														++nCameraNumber;
														break;

								default:				/* Unknown chunk */
														if (DIAGNOSTICS)
														{
															_RPT0(_CRT_WARN,  "Unknown NAMED_OBJECT chunk type.\n");
														}
								}
								break; /* NAMED_OBJECT */

		case MAT_ENTRY:			/* If chunk is a material entry, then save its file position */
								MatPos[nMatNumber] = FilePos;
								++nMatNumber;
								break; /* MAT_ENTRY */

		default:
			break;
		}

		/* Have we finished looking for chunks ? */
		FilePos += Size;

		if (FilePos > FileSize)
			break;

        /* Set file pointer to next chunk */
		pByte = p3DSFile + FilePos;
    }

	/* Write number of meshes, materials, lights and cameras used in object structure */
	nNumMesh		= nMeshNumber;
	nNumMaterial	= nMatNumber;
	nNumLight		= nLightNumber;
	nNumCamera		= nCameraNumber;

	/*********************
	** GEOMETRIC MESHES **
	*********************/

	/* Allocate memory for each mesh structure in the object */
	pMesh = new S3DSMesh[nNumMesh * sizeof(S3DSMesh)]; //(S3DSMesh *) calloc(nNumMesh, sizeof(S3DSMesh));

	if (!pMesh)
	{
		_RPT0(_CRT_WARN,  "Read3DSData : Not enough memory to allocate mesh structures\n");
		if (!bLoadFromResource) delete(p3DSFile);
		return false;
	}

#ifdef	FIXEDPOINTENABLE
	float * pFloatArray = 0;
#endif

	/* Reading data for each geometric Mesh */
	for (i = 0; i < nNumMesh; ++i)
	{
		pMesh[i].NodeID = -1;

		/* Go to next mesh file offset */
		FilePos = MeshPos[i];
		pByte = p3DSFile + FilePos;

		/* Read chunk and size */
		BREAD(&Chunk, pByte, sizeof(unsigned short));
		BREAD(&Size,  pByte, sizeof(unsigned int));
		ChunkSize = FilePos + Size;

		/* Increase file pointer by 6 (chunk+size) */
		FilePos += 6;

		/* Loop through each chunk until finished */
		while(true)
		{
			/* Read chunk and size */
			BREAD(&Chunk, pByte, sizeof(unsigned short));
			BREAD(&Size,  pByte, sizeof(unsigned int));

			/* Which type of chunk is it ? */
			switch (Chunk)
			{
			case POINT_ARRAY:	/* Read Number of vertices in mesh */
								BREAD(&pMesh[i].nNumVertex, pByte, sizeof(unsigned short));

								/* Allocate mesh memory for vertex geometry */
								//pMesh[i].pVertex  = (VERTTYPE *) malloc(pMesh[i].nNumVertex * 3 * sizeof(VERTTYPE));
								//pMesh[i].pNormals = (VERTTYPE *) malloc(pMesh[i].nNumVertex * 3 * sizeof(VERTTYPE));
								pMesh[i].pVertex  = new VERTTYPE[pMesh[i].nNumVertex * 3 * sizeof(VERTTYPE)];
								pMesh[i].pNormals = new VERTTYPE[pMesh[i].nNumVertex * 3 * sizeof(VERTTYPE)];

								/* Check that memory was correctly allocated */
								if (!pMesh[i].pVertex || !pMesh[i].pNormals)
								{
									_RPT0(_CRT_WARN,  "Model3DSRead : Not enough memory for object geometry\n");
									Destroy();

									if (!bLoadFromResource)
										delete(p3DSFile);

									return false;
								}

#ifdef	FIXEDPOINTENABLE	//If fixed point is enabled then convert the float values to fixed
								// pFloatArray  = (float *) malloc(pMesh[i].nNumVertex * 3 * sizeof(float));
								pFloatArray  =  new float *[pMesh[i].nNumVertex * 3 * sizeof(float)];

								if(!pFloatArray)
								{
									_RPT0(_CRT_WARN,  "Model3DSRead : Not enough memory for object geometry\n");
									Destroy();

									if (!bLoadFromResource)
										delete(p3DSFile);

									return false;
								}
								/* Read mesh vertices */
								BREAD(pFloatArray, pByte, pMesh[i].nNumVertex * 3 * sizeof(float));

								for(j = 0; j < pMesh[i].nNumVertex * 3; ++j)
									pMesh[i].pVertex[j] = f2vt(pFloatArray[j]);

								delete(pFloatArray);
#else
								BREAD(pMesh[i].pVertex, pByte, pMesh[i].nNumVertex * 3 * sizeof(VERTTYPE));
#endif
								fX = pMesh[i].pVertex[0];
								break; /* POINT_ARRAY */

            case FACE_ARRAY:	/* Read number of faces in mesh */
								BREAD(&pMesh[i].nNumFaces, pByte, sizeof(unsigned short));

								/* Allocate mesh memory for face list data */
								//pMesh[i].pFaces = (unsigned short *) malloc(pMesh[i].nNumFaces * 3 * sizeof(unsigned short));
								pMesh[i].pFaces = new unsigned short[pMesh[i].nNumFaces * 3 * sizeof(unsigned short)];

								/* Check that memory was correctly allocated */
								if(!pMesh[i].pFaces)
								{
									_RPT0(_CRT_WARN,  "Not enough memory for face list\n");
									//Model3DSDestroy(pObject);
									Destroy();

									if (!bLoadFromResource)
										delete(p3DSFile);

									return false;
								}

								/* Read face list information (we only take the first three parameters, as the fourth one is not needed) */
								for(j = 0; j < pMesh[i].nNumFaces; ++j)
								{
									/* Read indices in opposite order */
									BREAD(&pMesh[i].pFaces[j*3+2], pByte, sizeof(unsigned short));
									BREAD(&pMesh[i].pFaces[j*3+1], pByte, sizeof(unsigned short));
									BREAD(&pMesh[i].pFaces[j*3+0], pByte, sizeof(unsigned short));
									BREAD(&TmpShort, pByte, sizeof(short));
								}

								/* Read chunk and size */
								BREAD(&Chunk,     pByte, sizeof(unsigned short));
								BREAD(&TempSize,  pByte, sizeof(unsigned int));

								/* Assign material for this mesh to the first material found */
								if (Chunk==MSH_MAT_GROUP)
								{
									/* Read material name */
									for (j=0; j<16; j++)
									{
										BREAD(&pMesh[i].pszMaterial[j], pByte, sizeof(char));
										if (pMesh[i].pszMaterial[j]==0) break;
										if (pMesh[i].pszMaterial[j]<'0' || pMesh[i].pszMaterial[j]>'z')
										{
										//	pMesh[i].pszMaterial[j] = '_';
										}
									}
								}
								break; /* FACE_ARRAY */

			case TEX_VERTS:		/* Read number of vertices in mesh */
								BREAD(&pMesh[i].nNumVertex, pByte, sizeof(unsigned short));

								/* Allocate mesh memory for UVs */
								//pMesh[i].pUV = (VERTTYPE*) malloc(pMesh[i].nNumVertex * 2 * sizeof(VERTTYPE));
								pMesh[i].pUV = new VERTTYPE[pMesh[i].nNumVertex * 2 * sizeof(VERTTYPE)];

								/* Check that memory was correctly allocated */
								if (!pMesh[i].pUV)
								{
									_RPT0(_CRT_WARN,  "Not enough memory for UV list\n");
									Destroy();

									if (!bLoadFromResource)
										delete(p3DSFile);

									return false;
								}

#ifdef	FIXEDPOINTENABLE	//If fixed point is enabled then convert the float values to fixed
								// pFloatArray  = (float *) malloc(pMesh[i].nNumVertex * 3 * sizeof(float));
								pFloatArray  = new float *[pMesh[i].nNumVertex * 3 * sizeof(float)];

								if(!pFloatArray)
								{
									_RPT0(_CRT_WARN,  "Not enough memory for UV list\n");
									Destroy();

									if (!bLoadFromResource)
										delete(p3DSFile);

									return false;
								}

								/* Read UV coordinates for mesh */
								BREAD(pFloatArray, pByte, pMesh[i].nNumVertex * 2 * sizeof(float));

								for(j = 0; j < pMesh[i].nNumVertex * 2; ++j)
									pMesh[i].pUV[j] = f2vt(pFloatArray[j]);

								delete(pFloatArray);
#else
								/* Read UV coordinates for mesh */
								BREAD(pMesh[i].pUV, pByte, pMesh[i].nNumVertex * 2 * sizeof(float));
#endif
								break; /* TEX_VERTS */

			case MESH_MATRIX:	/* Read matrix information for object (not exported) */
								MatrixIdentity(g_Matrix[i]);

								BREAD(&TempFloat[0], pByte, sizeof(float) * 3);
								g_Matrix[i].f[ 0] = f2vt(TempFloat[0]);
								g_Matrix[i].f[ 4] = f2vt(TempFloat[1]);
								g_Matrix[i].f[ 8] = f2vt(TempFloat[2]);

								BREAD(&TempFloat[0], pByte, sizeof(float) * 3);
								g_Matrix[i].f[ 1] = f2vt(TempFloat[0]);
								g_Matrix[i].f[ 5] = f2vt(TempFloat[1]);
								g_Matrix[i].f[ 9] = f2vt(TempFloat[2]);

								BREAD(&TempFloat[0], pByte, sizeof(float) * 3);
								g_Matrix[i].f[ 2] = f2vt(TempFloat[0]);
								g_Matrix[i].f[ 6] = f2vt(TempFloat[1]);
								g_Matrix[i].f[10] = f2vt(TempFloat[2]);

								BREAD(&TempFloat[0], pByte, sizeof(float) * 3);
								g_Matrix[i].f[ 3] = f2vt(TempFloat[0]);
								g_Matrix[i].f[ 7] = f2vt(TempFloat[1]);
								g_Matrix[i].f[11] = f2vt(TempFloat[2]);
								break; /* MESH_MATRIX */

			default:			if (DIAGNOSTICS)
								{
									_RPT0(_CRT_WARN,  "Unknown N_TRI_OBJECT mesh chunk.\n");
								}
            }

			/* Have we finished looking for chunks in N_TRI_OBJECT ? */
			FilePos += Size;

			if (FilePos > ChunkSize)
				break;

			/* Set file to next chunk */
			pByte = p3DSFile + FilePos;
		}
	}

	/**************
	** MATERIALS **
	**************/

	/* Allocate memory for material structures in object */
	pMaterial = new S3DSMaterial[nNumMaterial * sizeof(S3DSMaterial)];// (S3DSMaterial *)calloc(nNumMaterial, sizeof(S3DSMaterial));

	/* Check that memory was correctly allocated */
	if(!pMaterial)
	{
		_RPT0(_CRT_WARN,  "Not enough memory for face list\n");
		Destroy();

		if (!bLoadFromResource)
			delete(p3DSFile);

		return false;
	}

	VERTTYPE fDiv;

	/* Read materials */
	for(i = 0; i < nNumMaterial; ++i)
	{
		/* Go to next material file offset */
		FilePos = MatPos[i];
		pByte = p3DSFile + FilePos;

		/* Read chunk and size */
		BREAD(&Chunk, pByte, sizeof(unsigned short));
		BREAD(&Size, pByte, sizeof(unsigned int));
		ChunkSize = FilePos + Size;

		/* Increase file pointer by 6 (chunk+size) */
		FilePos += 6;

		while (1)
		{
			/* Read chunk and size */
			BREAD(&Chunk, pByte, sizeof(unsigned short));
			BREAD(&Size, pByte, sizeof(unsigned int));

            switch (Chunk)
			{
			case MAT_NAME:		/* Read material name (16 characters plus NULL) */
								for (j=0; j<16; j++)
								{
									BREAD(&pMaterial[i].pszMatName[j], pByte, sizeof(char));
									if (pMaterial[i].pszMatName[j]==0) break;
								}
								break;

            case MAT_TEXMAP:	/* Read texture map strength (percentage) */
								BREAD(&Chunk, pByte, sizeof(unsigned short));
								if (Chunk==INT_PERCENTAGE) pByte+=12;
									else pByte+=20;

								/* Read texture map name */
								for (j=0; j<20; j++)
								{
									BREAD(&pMaterial[i].pszMatFile[j], pByte, 1);

									if(pMaterial[i].pszMatFile[j]==0)
										break;
								}
								break;

			case MAT_OPACMAP:	/* Read opacity map strength (percentage) */
								pMaterial[i].nMatOpacity = true;
								BREAD(&Chunk, pByte, sizeof(unsigned short));

								if (Chunk==INT_PERCENTAGE) pByte+=12;
									else pByte+=20;

								/* Read opacity map name */
								for (j=0; j<20; j++)
								{
									BREAD(&pMaterial[i].pszMatOpaq[j], pByte, sizeof(char));

									if (pMaterial[i].pszMatOpaq[j]==0)
										break;
								}
								break;

            case MAT_AMBIENT:	/* Read ambient material color */
								BREAD(&Chunk, pByte, sizeof(unsigned short));
								BREAD(&TempSize, pByte, sizeof(unsigned int));

								if (Chunk == COLOR_24)
								{
									fDiv = VERTTYPEDIV(1.0f, 255.0f);
									BREAD(&TempByte[0], pByte, 3*sizeof(char));
									pMaterial[i].fMatAmbient[0] = VERTTYPEMUL(f2vt((VERTTYPE) TempByte[0]), fDiv);
									pMaterial[i].fMatAmbient[1] = VERTTYPEMUL(f2vt((VERTTYPE) TempByte[1]), fDiv);
									pMaterial[i].fMatAmbient[2] = VERTTYPEMUL(f2vt((VERTTYPE) TempByte[2]), fDiv);
								}
								else if (Chunk == COLOR_F)
								{
									BREAD(&TempFloat[0], pByte, 3 * sizeof(float));
									pMaterial[i].fMatAmbient[0] = f2vt(TempFloat[0]);
									pMaterial[i].fMatAmbient[1] = f2vt(TempFloat[1]);
									pMaterial[i].fMatAmbient[2] = f2vt(TempFloat[2]);
								}
								break;

            case MAT_DIFFUSE:	/* Read diffuse material color */
								BREAD(&Chunk, pByte, sizeof(unsigned short));
								BREAD(&TempSize, pByte, sizeof(unsigned int));
								if (Chunk == COLOR_24)
								{
									fDiv = VERTTYPEDIV(1.0f, 255.0f);
									BREAD(&TempByte[0], pByte, 3*sizeof(char));
									pMaterial[i].fMatDiffuse[0] = VERTTYPEMUL(f2vt((VERTTYPE) TempByte[0]), fDiv);
									pMaterial[i].fMatDiffuse[1] = VERTTYPEMUL(f2vt((VERTTYPE) TempByte[1]), fDiv);
									pMaterial[i].fMatDiffuse[2] = VERTTYPEMUL(f2vt((VERTTYPE) TempByte[2]), fDiv);
								}
								else if (Chunk == COLOR_F)
								{
									BREAD(&TempFloat[0], pByte, 3 * sizeof(float));
									pMaterial[i].fMatDiffuse[0] = f2vt(TempFloat[0]);
									pMaterial[i].fMatDiffuse[1] = f2vt(TempFloat[1]);
									pMaterial[i].fMatDiffuse[2] = f2vt(TempFloat[2]);
								}
								break;

            case MAT_SPECULAR:	/* Read specular material color */
								BREAD(&Chunk, pByte, sizeof(unsigned short));
								BREAD(&TempSize, pByte, sizeof(unsigned int));
								if (Chunk == COLOR_24)
								{
									fDiv = VERTTYPEDIV(1.0f, 255.0f);
									BREAD(&TempByte[0], pByte, 3*sizeof(char));
									pMaterial[i].fMatSpecular[0] = VERTTYPEMUL(f2vt((VERTTYPE) TempByte[0]), fDiv);
									pMaterial[i].fMatSpecular[1] = VERTTYPEMUL(f2vt((VERTTYPE) TempByte[1]), fDiv);
									pMaterial[i].fMatSpecular[2] = VERTTYPEMUL(f2vt((VERTTYPE) TempByte[2]), fDiv);
								}
								else if (Chunk == COLOR_F)
								{
									BREAD(&TempFloat[0], pByte, 3 * sizeof(float));
									pMaterial[i].fMatSpecular[0] = f2vt(TempFloat[0]);
									pMaterial[i].fMatSpecular[1] = f2vt(TempFloat[1]);
									pMaterial[i].fMatSpecular[2] = f2vt(TempFloat[2]);
								}
								break;

            case MAT_SHININESS:	/* Read shininess ratio (percentage) */
								BREAD(&Chunk, pByte, sizeof(unsigned short));
								BREAD(&TempSize, pByte, sizeof(unsigned int));
								if (Chunk == INT_PERCENTAGE)
								{
									BREAD(&TmpShort, pByte, sizeof(unsigned short));
									pMaterial[i].fMatShininess = f2vt((float) TmpShort * 0.01f);
								}
								else if (Chunk == FLOAT_PERCENTAGE)
								{
									BREAD(&TempFloat[0], pByte, sizeof(float));
									pMaterial[i].fMatShininess = f2vt(TempFloat[0]);
								}
								break;

            case MAT_TRANSPARENCY:	/* Read material transparency */
									BREAD(&Chunk, pByte, sizeof(unsigned short));
									BREAD(&TempSize, pByte, sizeof(unsigned int));
									if (Chunk == INT_PERCENTAGE)
									{
										BREAD(&TmpShort, pByte, sizeof(unsigned short));
										pMaterial[i].fMatTransparency = f2vt((float) TmpShort * 0.01f);
									}
									else if (Chunk == FLOAT_PERCENTAGE)
									{
										BREAD(&TempFloat[0], pByte, sizeof(float));
										pMaterial[i].fMatTransparency = f2vt(TempFloat[0]);
									}
									break;

            case MAT_SHADING:		/* Material shading method */
									BREAD(&pMaterial[i].sMatShading, pByte, sizeof(short));
									break;

			default:				if (DIAGNOSTICS)
									{
										_RPT0(_CRT_WARN,  "Unknown MAT_ENTRY chunk.\n");
									}
			}

			/* Have we finished looking for chunks in MAT_ENTRY ? */
			FilePos += Size;
			if (FilePos > ChunkSize) break;
			pByte = p3DSFile + FilePos;
		}
	}

	/***********
	** LIGHTS **
	***********/
	if (nNumLight)
	{
		/* Allocate memory for all lights defined in object */
		pLight= new S3DSLight[nNumLight * sizeof(S3DSLight)]; // (S3DSLight *)calloc(nNumLight, sizeof(S3DSLight));
		if (!pLight)
		{
			_RPT0(_CRT_WARN,  "Model3DSRead : Not enough memory to allocate light structures\n");
			Destroy();

			if (!bLoadFromResource)
				delete(p3DSFile);

			return false;
		}

		/* Read data for each light */
		for (i=0; i < nNumLight; i++)
		{
			/* Go to next light file offset */
			FilePos = LightPos[i];
			pByte = p3DSFile + FilePos;

			/* Read chunk and size */
			BREAD(&Chunk, pByte, sizeof(unsigned short));
			BREAD(&Size, pByte, sizeof(unsigned int));

			/* Read light position (inverting y and z) */
			BREAD(&TempFloat[0], pByte, sizeof(float) * 3);

			pLight[i].fPosition.x = f2vt(TempFloat[0]);
			pLight[i].fPosition.y = f2vt(TempFloat[2]);
			pLight[i].fPosition.z = f2vt(TempFloat[1]);

			/* Read light color */
			BREAD(&Chunk, pByte, sizeof(unsigned short));
			BREAD(&TempSize, pByte, sizeof(unsigned int));

			if (Chunk == COLOR_24)
			{
				fDiv = VERTTYPEDIV(1.0f, 255.0f);
				BREAD(&TempByte[0], pByte, 3*sizeof(char));
				pLight[i].fColour[0] = f2vt((TempByte[0]) * fDiv);
				pLight[i].fColour[1] = f2vt((TempByte[1]) * fDiv);
				pLight[i].fColour[2] = f2vt((TempByte[2]) * fDiv);
			}
			else if (Chunk == COLOR_F)
			{
				BREAD(&TempFloat, pByte, 3 *sizeof(float));

				pLight[i].fColour[0] = f2vt(TempFloat[0]);
				pLight[i].fColour[1] = f2vt(TempFloat[1]);
				pLight[i].fColour[2] = f2vt(TempFloat[2]);
			}
		}
	}

	/************
	** CAMERAS **
	************/
	if (nNumCamera)
	{
		/* Allocate memory for all cameras defined in object */
		pCamera= new S3DSCamera[nNumCamera * sizeof(S3DSCamera)]; //(S3DSCamera *)calloc(nNumCamera, sizeof(S3DSCamera));
		if (!pCamera)
		{
			_RPT0(_CRT_WARN,  "Model3DSRead : Not enough memory to allocate camera structures\n");
			Destroy();

			if (!bLoadFromResource)
				delete(p3DSFile);
			return false;
		}

		/* Read data for each camera */
		for (i = 0; i < nNumCamera; i++)
		{
			/* Go to next camera file offset */
			FilePos = CameraPos[i];
			pByte = p3DSFile + FilePos;

			/* Read chunk and size */
			BREAD(&Chunk, pByte, sizeof(unsigned short));
			BREAD(&Size, pByte, sizeof(unsigned int));

			/* Read camera position (inverting y and z) */
			BREAD(&TempFloat[0], pByte, sizeof(float) * 3);

			pCamera[i].fPosition.x = f2vt(TempFloat[0]);
			pCamera[i].fPosition.y = f2vt(TempFloat[2]);
			pCamera[i].fPosition.z = f2vt(TempFloat[1]);

			/* Read camera looking point (inverting y and z) */
			BREAD(&TempFloat[0], pByte, sizeof(float) * 3);

			pCamera[i].fTarget.x = f2vt(TempFloat[0]);
			pCamera[i].fTarget.y = f2vt(TempFloat[2]);
			pCamera[i].fTarget.z = f2vt(TempFloat[1]);

			/* Read camera roll value */
			BREAD(&TempFloat[0], pByte, sizeof(float));
			pCamera[i].fRoll = f2vt(TempFloat[0]);

			/* Read camera focal length value */
			BREAD(&TempFloat[0], pByte, sizeof(float));
			pCamera[i].fFocalLength = f2vt(TempFloat[0]);

			/* Calculate FOV from focal length (FOV  = arctan(1/(2*f)))*/
			if(pCamera[i].fFocalLength == 0.0f)
			{
				pCamera[i].fFOV = f2vt(0.0f);
			}
			else
			{
				pCamera[i].fFOV = f2vt(70.0f) * f2vt((float) atan(1.0f / (2.0f * vt2f(pCamera[i].fFocalLength))));
			}

			/* Copy camera name */
			strcpy(pCamera[i].pszName, CameraName[i]);
		}
	}


	/*******************************
	** Feeding the Info Structure **
	*******************************/
	for(i = 0; i < nNumMesh; i++)
	{
		/* Copy mesh and material names */
		strcpy(pMesh[i].pszName, MeshName[i]);

		/* Find the material number corresponding to mesh */
		if(pMesh[i].pszMaterial)
		{
			/* Loop through all materials */
			for(j = 0; j < nNumMaterial; j++)
			{
				if (strcmp(pMesh[i].pszMaterial, pMaterial[j].pszMatName)==0)
				{
					pMesh[i].nIdxMaterial=j;
				}
			}
		}
	}

	/****************
	*****************
	** Read KFDATA **
	** Animation.  **
	*****************
	****************/

	/* Only read animation if it exists */
	if (KFDATAFileOffset!=-1)
	{
		char			pszName[20];
		unsigned int	TempFilePos;
		unsigned int	TmpInt;
		short			Revision;
		short			NodeId=0;
		short			SplineTerms;
		float			fDummy; //A dummy float used to skip the spline data

		/* First find out how many nodes are contained in the file */

		/* Reset file position */
		SecCont = 0;
		FilePos = KFDATAFileOffset + 6;

		/* Reset number of nodes */
		nNumNodes=0;

		/* Look for chunks */
		while (FilePos<FileSize && ++SecCont<MAX_MESHES)
		{
			/* Set file pointer to next chunk */
			pByte = p3DSFile + FilePos;

			/* Read chunk and size */
			BREAD(&Chunk, pByte, sizeof(unsigned short));
			BREAD(&Size, pByte, sizeof(unsigned int));

			/* Find object chunk */
			switch (Chunk)
			{
				case OBJECT_NODE_TAG:
				case CAMERA_NODE_TAG:
				case TARGET_NODE_TAG:
				case LIGHT_NODE_TAG:
				case SPOTLIGHT_NODE_TAG:
				case L_TARGET_NODE_TAG:		nNumNodes++;
			}

			/* Have we finished looking for chunks ? */
			FilePos += Size;
		}

		/* Debug info */
		if (DIAGNOSTICS)
		{
			sprintf(pszTmp, "Number of nodes = %d\n", nNumNodes);
			_RPT0(_CRT_WARN,  pszTmp);
		}

		/* Allocate memory for nodes */
		pNode = new S3DSNode[nNumNodes * sizeof(S3DSNode)]; //(S3DSNode *)calloc(nNumNodes, sizeof(S3DSNode));
		if (!pNode)
		{
			Destroy();

			if (!bLoadFromResource)
				delete(p3DSFile);

			return false;
		}

		/* Initialise nodes */
		for (i=0; i<nNumNodes; i++)
		{
			pNode[i].ParentIndex=-1;
		}


		/* Set pointer to start of data */
		SecCont = 0;

		FilePos = KFDATAFileOffset + 6;

		/*  Looking for sub-chunks */
		while(FilePos<FileSize && ++SecCont<MAX_MESHES)
		{
			/* Set file pointer to next chunk */
			pByte = p3DSFile + FilePos;

			/* Read chunk and size */
			BREAD(&Chunk, pByte, sizeof(unsigned short));
			BREAD(&Size, pByte, sizeof(unsigned int));

			switch(Chunk)
			{
			case KFHDR:	/* Keyframe header */

						/* Read revision number */
						BREAD(&Revision, pByte, sizeof(short));

						/* Read 3DS file name */
						for (j=0; j<20; j++)
						{
							BREAD(&pszName[j], pByte, sizeof(char));
							if(pszName[j] == 0) break;
							//if(pszName[j]<'0' || pszName[j]>'z') pszName[j] = '_';
							//if(pszName[0]>='0' && pszName[0]<='9') pszName[0] = 'N';
						}

						/* Read number of frames */
						BREAD(&nNumFrames, pByte, sizeof(unsigned int));

						/* Seems like frame 0 is not taken into account */
						nNumFrames++;

						/* Debug info */
						if (DIAGNOSTICS)
						{
							sprintf(pszTmp, "Animation header :\nRevision=%d\nName=%s\nNb of frames=%d\n", Revision, pszName, nNumFrames);
							_RPT0(_CRT_WARN,  pszTmp);
						}
						break;

			case KFSEG: /* Active frame segment */
						if (DIAGNOSTICS)
						{
							/* Keyframe active segment of frames to render */
							int nFirstFrame, nLastFrame;

							/* Read first and last frame */
							BREAD(&nFirstFrame, pByte, sizeof(unsigned int));
							BREAD(&nLastFrame, pByte, sizeof(unsigned int));

							/* Debug info */
							sprintf(pszTmp, "Frames to render %d -> %d\n", nFirstFrame, nLastFrame);
							_RPT0(_CRT_WARN,  pszTmp);
						}
						break;

			case KFCURTIME: /* Current frame number */
							if (DIAGNOSTICS)
							{
								/* Current active frame number */
								int nFrameNumber;

								/* Read frame number */
								BREAD(&nFrameNumber, pByte, sizeof(unsigned int));

								/* Debug info */
								sprintf(pszTmp, "Active frame number : %d\n", nFrameNumber);
								_RPT0(_CRT_WARN,  pszTmp);
							}
							break;

			case OBJECT_NODE_TAG:	/* Object node */

									/* Set file position after chunk and size */
									TempFilePos = FilePos + 6;

									/* Read all subchunks of OBJECT_NODE_TAG */
									while (TempFilePos < FilePos+Size)
									{
										/* Go to next subchunk */
										pByte = p3DSFile + TempFilePos;

										/* Read chunk and size */
										BREAD(&TempChunk, pByte, sizeof(unsigned short));
										BREAD(&TempSize, pByte, sizeof(unsigned int));

										/* Process chunk */
										switch(TempChunk)
										{
										case NODE_ID:	/* NODE_ID subchunk */
														BREAD(&NodeId, pByte, sizeof(short));
														break;

										case NODE_HDR:	/* NODE_HDR subchunk */

														/* Read mesh name */
														for (j=0; j<20; j++)
														{
															BREAD(&pszName[j], pByte, sizeof(char));
															if(pszName[j] == 0) break;
														}

														/* Find mesh index corresponding to that node */
														for (i=0; i < nNumMesh; i++)
														{
															if(strcmp(pMesh[i].pszName, pszName)==0)
															{
																pMesh[i].NodeID = NodeId;
																break;
															}
														}

														/* Read flags (2 of them) (not used) */
														BREAD(&TmpShort, pByte, sizeof(short));
														BREAD(&TmpShort, pByte, sizeof(short));

														/* Parent index */
														BREAD(&pNode[NodeId].ParentIndex, pByte, sizeof(short));

														/* Debug info */
														if (DIAGNOSTICS)
														{
															sprintf(pszTmp, "------------\nNode ID = %d\n------------\nMesh name = %s\nParent index = %d\n", NodeId, pszName, pNode[NodeId].ParentIndex);
															_RPT0(_CRT_WARN,  pszTmp);
														}
														break;

										case PIVOT:		/* PIVOT subchunk */

														/* Read pivot point */
														BREAD(&TempFloat[0], pByte, sizeof(float) * 3);
														pNode[NodeId].Pivot.x = f2vt(TempFloat[0]);
														pNode[NodeId].Pivot.y = f2vt(TempFloat[1]);
														pNode[NodeId].Pivot.z = f2vt(TempFloat[2]);

														/* Debug info */
														if (DIAGNOSTICS)
														{
															sprintf(pszTmp, "Pivot point : %.2f %.2f %.2f\n", pNode[NodeId].Pivot.x, pNode[NodeId].Pivot.y, pNode[NodeId].Pivot.z);
															_RPT0(_CRT_WARN,  pszTmp);
														}
														break;

										case BOUNDBOX:	/* BOUNDBOX subchunk */
														if (DIAGNOSTICS)
														{
															float MinX, MinY, MinZ;
															float MaxX, MaxY, MaxZ;

															/* Read Minimum bounding point */
															BREAD(&MinX, pByte, sizeof(float));
															BREAD(&MinY, pByte, sizeof(float));
															BREAD(&MinZ, pByte, sizeof(float));

															/* Read Maximum bounding point */
															BREAD(&MaxX, pByte, sizeof(float));
															BREAD(&MaxY, pByte, sizeof(float));
															BREAD(&MaxZ, pByte, sizeof(float));

															/* Debug info */
															sprintf(pszTmp, "Bounding Box : Min=(%.2f, %.2f, %.2f)  Max=(%.2f, %.2f, %.2f)\n", MinX, MinY, MinZ, MaxX, MaxY, MaxZ);
															_RPT0(_CRT_WARN,  pszTmp);
														}
														break;

										case POS_TRACK_TAG: /* POS_TRACK_TAG subchunk */

															/* Read internal flags and 2 unused unsigned int */
															BREAD(&TmpShort, pByte, sizeof(short));
															BREAD(&TmpInt, pByte, sizeof(unsigned int));
															BREAD(&TmpInt, pByte, sizeof(unsigned int));

															/* Read number of keys in track */
															BREAD(&pNode[NodeId].PositionKeys, pByte, sizeof(unsigned int));

															/* Debug info */
															if (DIAGNOSTICS)
															{
																sprintf(pszTmp, "Position keys in track = %d\n", pNode[NodeId].PositionKeys);
																_RPT0(_CRT_WARN,  pszTmp);
															}

															/* Allocate memory for these keys */
															if (pNode[NodeId].PositionKeys)
															{
																pNode[NodeId].pPosition = new POSITIONKEY[pNode[NodeId].PositionKeys * sizeof(POSITIONKEY)]; //(POSITIONKEY *)calloc(pNode[NodeId].PositionKeys, sizeof(POSITIONKEY));
															}

															/* Read all keys */
															for (i=0; i<pNode[NodeId].PositionKeys; i++)
															{
																/* Read frame number */
																BREAD(&pNode[NodeId].pPosition[i].FrameNumber, pByte, sizeof(unsigned int));

																/* Read spline terms */
																BREAD(&SplineTerms, pByte, sizeof(short));

																if (SplineTerms & 0x01)
																{
																	BREAD(&TempFloat[0], pByte, sizeof(float));
																	pNode[NodeId].pPosition[i].fTension = f2vt(TempFloat[0]);
																}

																if (SplineTerms & 0x02)
																{
																	BREAD(&TempFloat[0], pByte, sizeof(float));
																	pNode[NodeId].pPosition[i].fContinuity = f2vt(TempFloat[0]);
																}

																if (SplineTerms & 0x04)
																{
																	BREAD(&TempFloat[0], pByte, sizeof(float));
																	pNode[NodeId].pPosition[i].fBias = f2vt(TempFloat[0]);
																}

																if (SplineTerms & 0x08)
																	BREAD(&fDummy, pByte, sizeof(float));
																if (SplineTerms & 0x10)
																	BREAD(&fDummy, pByte, sizeof(float));

																/* Read position */
																BREAD(&TempFloat[0], pByte, sizeof(float) * 3);

																pNode[NodeId].pPosition[i].p.x = f2vt(TempFloat[0]);
																pNode[NodeId].pPosition[i].p.y = f2vt(TempFloat[1]);
																pNode[NodeId].pPosition[i].p.z = f2vt(TempFloat[2]);

																/* Debug info */
																if (DIAGNOSTICS)
																{
																	sprintf(pszTmp, "Frame %d : Translation of (%.2f, %.2f, %.2f)\n", pNode[NodeId].pPosition[i].FrameNumber, pNode[NodeId].pPosition[i].p.x, pNode[NodeId].pPosition[i].p.y, pNode[NodeId].pPosition[i].p.z);
																	_RPT0(_CRT_WARN,  pszTmp);
																	sprintf(pszTmp, "Spline terms : (%.2f, %.2f, %.2f)\n", pNode[NodeId].pPosition[i].fTension, pNode[NodeId].pPosition[i].fContinuity, pNode[NodeId].pPosition[i].fBias);
																	_RPT0(_CRT_WARN,  pszTmp);
																}
															}
															break;

										case ROT_TRACK_TAG: /* ROT_TRACK_TAG subchunk */

															/* Read internal flags and 2 unused unsigned int */
															BREAD(&TmpShort, pByte, sizeof(short));
															BREAD(&TmpInt, pByte, sizeof(unsigned int));
															BREAD(&TmpInt, pByte, sizeof(unsigned int));

															/* Read number of keys in track */
															BREAD(&pNode[NodeId].RotationKeys, pByte, sizeof(unsigned int));

															/* Debug info */
															if (DIAGNOSTICS)
															{
																sprintf(pszTmp, "Rotation keys in track = %d\n", pNode[NodeId].RotationKeys);
																_RPT0(_CRT_WARN,  pszTmp);
															}

															/* Allocate memory for these keys */
															if (pNode[NodeId].RotationKeys)
															{
																pNode[NodeId].pRotation= new ROTATIONKEY[pNode[NodeId].RotationKeys * sizeof(ROTATIONKEY)]; //(ROTATIONKEY *)calloc(pNode[NodeId].RotationKeys, sizeof(ROTATIONKEY));
															}

															/* Read all keys */
															for (i=0; i<pNode[NodeId].RotationKeys; i++)
															{
																/* Read frame number */
																BREAD(&pNode[NodeId].pRotation[i].FrameNumber, pByte, sizeof(int));

																/* Read spline terms */
																BREAD(&SplineTerms, pByte, sizeof(short));

																if (SplineTerms & 0x01)
																{
																	BREAD(&TempFloat[0], pByte, sizeof(float));
																	pNode[NodeId].pRotation[i].fTension = f2vt(TempFloat[0]);
																}

																if (SplineTerms & 0x02)
																{
																	BREAD(&TempFloat[0], pByte, sizeof(float));
																	pNode[NodeId].pRotation[i].fContinuity = f2vt(TempFloat[0]);
																}

																if (SplineTerms & 0x04)
																{
																	BREAD(&TempFloat[0], pByte, sizeof(float));
																	pNode[NodeId].pRotation[i].fBias = f2vt(TempFloat[0]);
																}

																if (SplineTerms & 0x08) BREAD(&fDummy, pByte, sizeof(float));
																if (SplineTerms & 0x10) BREAD(&fDummy, pByte, sizeof(float));

																/* Read rotation values (first one is angle) */
																BREAD(&TempFloat[0], pByte, sizeof(float));
																pNode[NodeId].pRotation[i].Angle = f2vt(TempFloat[0]);

																BREAD(&TempFloat[0], pByte, sizeof(float) * 3);
																pNode[NodeId].pRotation[i].r.x = f2vt(TempFloat[0]);
																pNode[NodeId].pRotation[i].r.y = f2vt(TempFloat[1]);
																pNode[NodeId].pRotation[i].r.z = f2vt(TempFloat[2]);

																/* Debug info */
																if (DIAGNOSTICS)
																{
																	sprintf(pszTmp, "Frame %d : Rotation of %.2f radians around the (%.2f, %.2f, %.2f) axis\n", pNode[NodeId].pRotation[i].FrameNumber, pNode[NodeId].pRotation[i].Angle, pNode[NodeId].pRotation[i].r.x, pNode[NodeId].pRotation[i].r.y, pNode[NodeId].pRotation[i].r.z);
																	_RPT0(_CRT_WARN,  pszTmp);
																	sprintf(pszTmp, "Spline terms : (%.2f, %.2f, %.2f)\n", pNode[NodeId].pRotation[i].fTension, pNode[NodeId].pRotation[i].fContinuity, pNode[NodeId].pRotation[i].fBias);
																	_RPT0(_CRT_WARN,  pszTmp);
																}
															}
															break;

										case SCL_TRACK_TAG: /* SCL_TRACK_TAG subchunk */

															/* Read internal flags and 2 unused unsigned int */
															BREAD(&TmpShort, pByte, sizeof(short));
															BREAD(&TmpInt, pByte, sizeof(unsigned int));
															BREAD(&TmpInt, pByte, sizeof(unsigned int));

															/* Read number of keys in track */
															BREAD(&pNode[NodeId].ScalingKeys, pByte, sizeof(unsigned int));

															/* Debug info */
															if (DIAGNOSTICS)
															{
																sprintf(pszTmp, "Scaling keys in track = %d\n", pNode[NodeId].ScalingKeys);
																_RPT0(_CRT_WARN,  pszTmp);
															}

															/* Allocate memory for these keys */
															if (pNode[NodeId].ScalingKeys)
															{
																pNode[NodeId].pScaling= new SCALINGKEY[pNode[NodeId].ScalingKeys * sizeof(SCALINGKEY)]; //(SCALINGKEY *)calloc(pNode[NodeId].ScalingKeys, sizeof(SCALINGKEY));
															}

															/* Read all keys */
															for (i=0; i<pNode[NodeId].ScalingKeys; i++)
															{
																/* Read frame number */
																BREAD(&pNode[NodeId].pScaling[i].FrameNumber, pByte, sizeof(int));

																/* Read spline terms */
																BREAD(&SplineTerms, pByte, sizeof(short));

																if (SplineTerms & 0x01)
																{
																	BREAD(&TempFloat[0], pByte, sizeof(float));
																	pNode[NodeId].pScaling[i].fTension = f2vt(TempFloat[0]);
																}

																if (SplineTerms & 0x02)
																{
																	BREAD(&TempFloat[0], pByte, sizeof(float));
																	pNode[NodeId].pScaling[i].fContinuity = f2vt(TempFloat[0]);
																}

																if (SplineTerms & 0x04)
																{
																	BREAD(&TempFloat[0], pByte, sizeof(float));
																	pNode[NodeId].pScaling[i].fBias = f2vt(TempFloat[0]);
																}

																if (SplineTerms & 0x08) BREAD(&fDummy, pByte, sizeof(float));
																if (SplineTerms & 0x10) BREAD(&fDummy, pByte, sizeof(float));

																/* Read scaling values */
																BREAD(&TempFloat[0], pByte, sizeof(float) * 3);
																pNode[NodeId].pScaling[i].s.x = f2vt(TempFloat[0]);
																pNode[NodeId].pScaling[i].s.y = f2vt(TempFloat[1]);
																pNode[NodeId].pScaling[i].s.z = f2vt(TempFloat[2]);

																/* Debug info */
																if (DIAGNOSTICS)
																{
																	sprintf(pszTmp, "Frame %d : Scaling of (%.2f, %.2f, %.2f)\n", pNode[NodeId].pScaling[i].FrameNumber, pNode[NodeId].pScaling[i].s.x, pNode[NodeId].pScaling[i].s.y, pNode[NodeId].pScaling[i].s.z);
																	_RPT0(_CRT_WARN,  pszTmp);
																	sprintf(pszTmp, "Spline terms : (%.2f, %.2f, %.2f)\n", pNode[NodeId].pScaling[i].fTension, pNode[NodeId].pScaling[i].fContinuity, pNode[NodeId].pScaling[i].fBias);
																	_RPT0(_CRT_WARN,  pszTmp);
																}
															}
															break;

										case MORPH_TRACK_TAG:	/* Morph object keys (not exported) */
																if (DIAGNOSTICS)
																{
																	_RPT0(_CRT_WARN,  "MORPH_TRACK_TAG chunk found\n");
																}
																break;

										case MORPH_SMOOTH:		/* Smoothing angle for morphing objects */
																if (DIAGNOSTICS)
																{
																	_RPT0(_CRT_WARN,  "MORPH_SMOOTH chunk found\n");
																}
																break;

										case INSTANCE_NAME:		/* Mesh instance name */
																if (DIAGNOSTICS)
																{
																	char pszInstance[12];

																	_RPT0(_CRT_WARN,  "INSTANCE_NAME chunk found\n");
																	BREAD(pszInstance, pByte, 11*sizeof(char));
																	sprintf(pszTmp, "Instance name = %s\n", pszInstance);
																	_RPT0(_CRT_WARN,  pszTmp);
																}
																break;

										default:				/* Unknown chunks */
																if (DIAGNOSTICS)
																{
																	_RPT0(_CRT_WARN,  "Unknown OBJECT_NODE_TAG chunk\n");
																	sprintf(pszTmp ,"Unknown chunk is 0x%X\n", TempChunk);
																	_RPT0(_CRT_WARN,  pszTmp);
																}
										}

										/* Next subchunk offset */
										TempFilePos+=TempSize;
									}

									/* Increase current node index */
									NodeId++;
									break;

	case TARGET_NODE_TAG :	bNodeTarget = 1; // mark node as target

	case CAMERA_NODE_TAG:	/* Object node */

									/* Set file position after chunk and size */
									TempFilePos = FilePos + 6;

									/* Read all subchunks of OBJECT_NODE_TAG */
									while (TempFilePos < FilePos+Size)
									{
										/* Go to next subchunk */
										pByte = p3DSFile + TempFilePos;

										/* Read chunk and size */
										BREAD(&TempChunk, pByte, sizeof(unsigned short));
										BREAD(&TempSize, pByte, sizeof(unsigned int));

										/* Process chunk */
										switch(TempChunk)
										{
										case NODE_ID:	/* NODE_ID subchunk */
														BREAD(&NodeId, pByte, sizeof(short));
														break;


										case NODE_HDR:	/* NODE_HDR subchunk */

														/* Read mesh name */
														for (j = 0; j < 20; j++)
														{
															BREAD(&pszName[j], pByte, sizeof(char));

															if(pszName[j] == 0)
																break;
														}

														/* Find camera index corresponding to that node */
														for (i=0; i<nNumCamera; i++)
														{

															if (strcmp(pCamera[i].pszName, pszName)==0)
															{
																if(bNodeTarget==0)
																{
																	pCamera[i].NodeID = NodeId;
																	bNodeTarget = 0;
																}
																else
																{
																	pCamera[i].TargetNodeID = NodeId;
																	bNodeTarget = 0;
																}
																break;
															}
														}

														/* Read flags (2 of them) (not used) */
														BREAD(&TmpShort, pByte, sizeof(short));
														BREAD(&TmpShort, pByte, sizeof(short));

														/* Parent index */
														BREAD(&pNode[NodeId].ParentIndex, pByte, sizeof(short));

														/* Debug info */
														if (DIAGNOSTICS)
														{
															sprintf(pszTmp, "------------\nCamera ID = %d \n------------\nCamera name = %s\nParent index = %d\n", NodeId, pszName, pNode[NodeId].ParentIndex);
															_RPT0(_CRT_WARN,  pszTmp);
														}
														break;

										case POS_TRACK_TAG: /* P	Material[i].dwVertexShaderList	CXX0017: Error: symbol "Material" not found
OS_TRACK_TAG subchunk */
															_RPT0(_CRT_WARN,  "here POS_TRACK_TAG chunk\n");

															/* Read internal flags and 2 unused unsigned int */
															BREAD(&TmpShort, pByte, sizeof(short));
															BREAD(&TmpInt, pByte, sizeof(unsigned int));
															BREAD(&TmpInt, pByte, sizeof(unsigned int));

															/* Read number of keys in track */
															BREAD(&pNode[NodeId].PositionKeys, pByte, sizeof(unsigned int));

															/* Debug info */
															if (DIAGNOSTICS)
															{
																sprintf(pszTmp, "Position keys in track = %d\n", pNode[NodeId].PositionKeys);
																_RPT0(_CRT_WARN,  pszTmp);
															}

															/* Allocate memory for these keys */
															if (pNode[NodeId].PositionKeys)
															{
																pNode[NodeId].pPosition = new POSITIONKEY[pNode[NodeId].PositionKeys * sizeof(POSITIONKEY)]; //(POSITIONKEY *)calloc(pNode[NodeId].PositionKeys, sizeof(POSITIONKEY));
															}

															/* Read all keys */
															for (i=0; i<pNode[NodeId].PositionKeys; i++)
															{
																/* Read frame number */
																BREAD(&pNode[NodeId].pPosition[i].FrameNumber, pByte, sizeof(unsigned int));

																/* Read spline terms */
																BREAD(&SplineTerms, pByte, sizeof(short));

																if (SplineTerms & 0x01)
																{
																	BREAD(&TempFloat[0], pByte, sizeof(float));
																	pNode[NodeId].pPosition[i].fTension = f2vt(TempFloat[0]);
																}

																if (SplineTerms & 0x02)
																{
																	BREAD(&TempFloat[0], pByte, sizeof(float));
																	pNode[NodeId].pPosition[i].fContinuity = f2vt(TempFloat[0]);
																}

																if (SplineTerms & 0x04)
																{
																	BREAD(&TempFloat[0], pByte, sizeof(float));
																	pNode[NodeId].pPosition[i].fBias = f2vt(TempFloat[0]);
																}

																if (SplineTerms & 0x08) BREAD(&fDummy, pByte, sizeof(float));
																if (SplineTerms & 0x10) BREAD(&fDummy, pByte, sizeof(float));

																/* Read position */
																BREAD(&TempFloat[0], pByte, sizeof(float) * 3);
																pNode[NodeId].pPosition[i].p.x = f2vt(TempFloat[0]);
																pNode[NodeId].pPosition[i].p.y = f2vt(TempFloat[2]);
																pNode[NodeId].pPosition[i].p.z = f2vt(TempFloat[1]);

																/* Debug info */
																if(DIAGNOSTICS)
																{
																	sprintf(pszTmp, "Frame %d : Translation of (%.2f, %.2f, %.2f)\n", pNode[NodeId].pPosition[i].FrameNumber, vt2f(pNode[NodeId].pPosition[i].p.x), vt2f(pNode[NodeId].pPosition[i].p.y), vt2f(pNode[NodeId].pPosition[i].p.z));
																	_RPT0(_CRT_WARN,  pszTmp);
																	sprintf(pszTmp, "Spline terms : (%.2f, %.2f, %.2f)\n", vt2f(pNode[NodeId].pPosition[i].fTension), vt2f(pNode[NodeId].pPosition[i].fContinuity), vt2f(pNode[NodeId].pPosition[i].fBias));
																	_RPT0(_CRT_WARN,  pszTmp);
																}
															}
															break;


										default:				/* Unknown chunks */
																if(DIAGNOSTICS)
																{
																	_RPT0(_CRT_WARN,  "Unknown OBJECT_NODE_TAG chunk\n");
																	sprintf(pszTmp ,"Unknown chunk is 0x%X\n", TempChunk);
																	_RPT0(_CRT_WARN,  pszTmp);
																}
										}

										/* Next subchunk offset */
										TempFilePos+=TempSize;
									}

									/* Increase current node index */
									NodeId++;
									break;

			case LIGHT_NODE_TAG :
				NodeId++;

				if(DIAGNOSTICS)
				{
					_RPT0(_CRT_WARN,  "LIGHT_NODE_TAG found\n");
				}
				break;
			case SPOTLIGHT_NODE_TAG :
				NodeId++;

				if(DIAGNOSTICS)
				{
					_RPT0(_CRT_WARN,  "SPOTLIGHT_NODE_TAG found\n");
				}
				break;
			case L_TARGET_NODE_TAG :
				NodeId++;

				if(DIAGNOSTICS)
				{
					_RPT0(_CRT_WARN,  "L_TARGET_NODE_TAG found\n");
				}
				break;
			default:
				if(DIAGNOSTICS)
				{
					_RPT0(_CRT_WARN,  "Unknown KFDATA chunk found.\n");
				}
				break;
			}

			/* Have we finished looking for chunks ? */
			FilePos += Size;
		}
	}

	/* Done. We don't need the .3DS model anymore. */
	if (!bLoadFromResource) delete(p3DSFile);


	/********************************
	*********************************
	** PROCESS EACH MESH IN OBJECT **
	*********************************
	********************************/
	/* Meshes only need to be processed if animation data was found */
	if (KFDATAFileOffset!=-1)
	{
		/* Loop through each mesh */
		for (i=0; i<nNumMesh; i++)
		{
			MATRIX	FinalMatrix, TransMatrix, FlipX, ObjectOffsetMatrix;
			VECTOR3	V1, V2, V3, CrossProduct;
			unsigned short		Tmp;

			/******************************
			** Get World-to-Local Matrix **
			******************************/
			/* Get MESH_MATRIX for this mesh. This matrix is the transformation matrix
			   that transform a mesh from its local space to world space */
			TransMatrix = g_Matrix[i];

			/* Inverse it. This matrix can be used to transform the vertices from world space
			   to their local space (required for keyframer operations) */
			MatrixInverse(FinalMatrix, TransMatrix);


			/*****************
			** Parity Check **
			*****************/
			/* Check for objects that have been flipped: their 3D "parity" will be off */
			/* Get vectors from transformation matrix */
			V1.x = TransMatrix.f[ 0];	V1.y = TransMatrix.f[ 1];	V1.z = TransMatrix.f[ 2];
			V2.x = TransMatrix.f[ 4];	V2.y = TransMatrix.f[ 5];	V2.z = TransMatrix.f[ 6];
			V3.x = TransMatrix.f[ 8];	V3.y = TransMatrix.f[ 9];	V3.z = TransMatrix.f[10];

			/* Compute cross product vector between V1 and V2 */
			MatrixVec3CrossProduct(CrossProduct, V1, V2);

			/* If dot product is negative the object has been flipped */
			if (MatrixVec3DotProduct(CrossProduct, V3) < 0.0f)
			{
				/* Debug output */
				if (DIAGNOSTICS)
				{
					_RPT0(_CRT_WARN,  "Object has been flipped\n");
				}

				/* Flip the X coordinate by appending the FlipX matrix to our final matrix */
				MatrixIdentity(FlipX);
				FlipX.f[0] = f2vt(-1.0f);
				MatrixMultiply(FinalMatrix, FinalMatrix, FlipX);

				/* Flip triangle ordering since we just flipped X */
				for (j=0; j<3*pMesh[i].nNumFaces; j+=3)
				{
					Tmp = pMesh[i].pFaces[j + 0];
					pMesh[i].pFaces[j + 0] = pMesh[i].pFaces[j + 2];
					pMesh[i].pFaces[j + 2] = Tmp;
				}
			}

			/**************************************************************
			** Compute Local-to-World Matrix Using Keyframer Information **
			**************************************************************/
			if(pMesh[i].NodeID != -1)
			{
				/* Reset matrix */
				MatrixIdentity(ObjectOffsetMatrix);

				/* Subtract pivot point from the matrix */
				ObjectOffsetMatrix.f[ 3] -= pNode[pMesh[i].NodeID].Pivot.x;
				ObjectOffsetMatrix.f[ 7] -= pNode[pMesh[i].NodeID].Pivot.y;
				ObjectOffsetMatrix.f[11] -= pNode[pMesh[i].NodeID].Pivot.z;

				/* Get local-to-world matrix for this mesh at frame 0 */
				ObjectOffsetMatrix = GetHierarchyMatrix(pMesh[i].NodeID, 0, ObjectOffsetMatrix);

				/* Concatenate matrices together */
				MatrixMultiply(FinalMatrix, FinalMatrix, ObjectOffsetMatrix);
			}

			/************************************************
			** Append Inversion Matrix (Inverting Y and Z) **
			************************************************/
			memset(&InversionMatrix, 0, sizeof(MATRIX));

			InversionMatrix.f[ 0]= f2vt(1.0f);
			InversionMatrix.f[ 6]= f2vt(1.0f);	// TMP
			InversionMatrix.f[ 9]= f2vt(1.0f);
			InversionMatrix.f[15]= f2vt(1.0f);

			/* Concatenate matrices together */
			MatrixMultiply(FinalMatrix, FinalMatrix, InversionMatrix);


			/*****************************************
			** Transform Vertices With Final Matrix **
			*****************************************/
			/* Get source pointer */
			pSource = (VECTOR3 *)pMesh[i].pVertex;

			/* Transform all vertices with FinalMatrix */
			for (j=0; j<pMesh[i].nNumVertex; j++)
			{
				/* Compute transformed vertex */
				D.x =	VERTTYPEMUL(pSource->x, FinalMatrix.f[ 0]) +
						VERTTYPEMUL(pSource->y, FinalMatrix.f[ 4]) +
						VERTTYPEMUL(pSource->z, FinalMatrix.f[ 8]) +
						FinalMatrix.f[12];

				D.y =	VERTTYPEMUL(pSource->x, FinalMatrix.f[ 1]) +
						VERTTYPEMUL(pSource->y, FinalMatrix.f[ 5]) +
						VERTTYPEMUL(pSource->z, FinalMatrix.f[ 9]) +
						FinalMatrix.f[13];

				D.z =	VERTTYPEMUL(pSource->x, FinalMatrix.f[ 2]) +
						VERTTYPEMUL(pSource->y, FinalMatrix.f[ 6]) +
						VERTTYPEMUL(pSource->z, FinalMatrix.f[10]) +
						FinalMatrix.f[14];

				/* Replace vertex with transformed vertex */
				*pSource++ = D;
			}
		}
	}
	else
	{
		/***************************************
		** NO ANIMATION EXPORTED (OLD MODELS) **
		***************************************/

		/* Make sure we set the number of nodes and frames to 0 */
		nNumNodes  = 0;
		nNumFrames = 0;
		pNode =		NULL;

		/* Create inversion matrix */
		memset(&InversionMatrix, 0, sizeof(MATRIX));

		InversionMatrix.f[ 0]= f2vt(1.0f);
		InversionMatrix.f[ 6]= f2vt(1.0f);
		InversionMatrix.f[ 9]= f2vt(1.0f);
		InversionMatrix.f[15]= f2vt(1.0f);

		/* Loop through each mesh */
		for (i=0; i<nNumMesh; i++)
		{
			/*********************************************
			** Transform Vertices with Inversion Matrix **
			*********************************************/
			/* Get source pointer */
			pSource = (VECTOR3 *) pMesh[i].pVertex;

			/* Transform all vertices with FinalMatrix */
			for (j=0; j < pMesh[i].nNumVertex; j++)
			{
				/* Compute transformed vertex */
				D.x =	VERTTYPEMUL(pSource->x, InversionMatrix.f[ 0]) +
						VERTTYPEMUL(pSource->y, InversionMatrix.f[ 4]) +
						VERTTYPEMUL(pSource->z, InversionMatrix.f[ 8]) +
						InversionMatrix.f[12];

				D.y =	VERTTYPEMUL(pSource->x, InversionMatrix.f[ 1]) +
						VERTTYPEMUL(pSource->y, InversionMatrix.f[ 5]) +
						VERTTYPEMUL(pSource->z, InversionMatrix.f[ 9]) +
						InversionMatrix.f[13];

				D.z =	VERTTYPEMUL(pSource->x, InversionMatrix.f[ 2]) +
						VERTTYPEMUL(pSource->y, InversionMatrix.f[ 6]) +
						VERTTYPEMUL(pSource->z, InversionMatrix.f[10]) +
						InversionMatrix.f[14];

				/* Replace vertex with transformed vertex */
				*pSource++ = D;
			}
		}
	}


	/********************************
	** Compute normals and centres **
	********************************/
	/* Loop through each mesh */
	for (i=0; i<nNumMesh; i++)
	{
		/********************
		** Compute Normals **
		********************/
		CalculateNormals(pMesh[i].nNumVertex, pMesh[i].pVertex,
						 pMesh[i].nNumFaces, pMesh[i].pFaces,
						 pMesh[i].pNormals);

		/************************
		** Compute mesh centre **
		************************/
 		/* Initialise boundary box values */
		pMesh[i].fMinimum.x = pMesh[i].fMaximum.x = pMesh[i].pVertex[0];
		pMesh[i].fMinimum.y = pMesh[i].fMaximum.y = pMesh[i].pVertex[1];
		pMesh[i].fMinimum.z = pMesh[i].fMaximum.z = pMesh[i].pVertex[2];

		/* For each vertex of each mesh */
		for(j = 0; j < 3 * pMesh[i].nNumVertex; j += 3)
		{
			/* Get current vertex */
			fX = pMesh[i].pVertex[j + 0];
			fY = pMesh[i].pVertex[j + 1];
			fZ = pMesh[i].pVertex[j + 2];

			/* Mesh minimum */
			if (fX < pMesh[i].fMinimum.x)	pMesh[i].fMinimum.x = fX;
			if (fY < pMesh[i].fMinimum.y)	pMesh[i].fMinimum.y = fY;
			if (fZ < pMesh[i].fMinimum.z)	pMesh[i].fMinimum.z = fZ;

			/* Mesh maximum */
			if (fX > pMesh[i].fMaximum.x)	pMesh[i].fMaximum.x = fX;
			if (fY > pMesh[i].fMaximum.y)	pMesh[i].fMaximum.y = fY;
			if (fZ > pMesh[i].fMaximum.z)	pMesh[i].fMaximum.z = fZ;
		}

		/* Write mesh centre */
		pMesh[i].fCentre.x = VERTTYPEMUL((pMesh[i].fMinimum.x + pMesh[i].fMaximum.x), f2vt(0.5f));
		pMesh[i].fCentre.y = VERTTYPEMUL((pMesh[i].fMinimum.y + pMesh[i].fMaximum.y), f2vt(0.5f));
		pMesh[i].fCentre.z = VERTTYPEMUL((pMesh[i].fMinimum.z + pMesh[i].fMaximum.z), f2vt(0.5f));
	}


	/******************************************
	** Calculating the total number of polys **
	******************************************/
	nTotalVertices = 0;
	nTotalFaces = 0;

	for (i = 0; i < nNumMesh; ++i)
	{
		nTotalVertices += pMesh[i].nNumVertex;
		nTotalFaces += pMesh[i].nNumFaces;
	}


	/*********************************
	** Compute object global centre **
	*********************************/
	/* Initialise bounding box values */
	fGroupMinimum.x = fGroupMaximum.x = pMesh[0].fMinimum.x;
	fGroupMinimum.y = fGroupMaximum.y = pMesh[0].fMinimum.y;
	fGroupMinimum.z = fGroupMaximum.z = pMesh[0].fMinimum.z;

	/* Look through all meshes */
	for (i = 0; i < nNumMesh; ++i)
	{
		if (pMesh[i].fMinimum.x < fGroupMinimum.x) fGroupMinimum.x = pMesh[i].fMinimum.x;
		if (pMesh[i].fMinimum.y < fGroupMinimum.y) fGroupMinimum.y = pMesh[i].fMinimum.y;
		if (pMesh[i].fMinimum.z < fGroupMinimum.z) fGroupMinimum.z = pMesh[i].fMinimum.z;

		if (pMesh[i].fMaximum.x > fGroupMaximum.x) fGroupMaximum.x = pMesh[i].fMaximum.x;
		if (pMesh[i].fMaximum.y > fGroupMaximum.y) fGroupMaximum.y = pMesh[i].fMaximum.y;
		if (pMesh[i].fMaximum.z > fGroupMaximum.z) fGroupMaximum.z = pMesh[i].fMaximum.z;
	}

	/* Finally compute object centre from bounding box */
	fGroupCentre.x = VERTTYPEMUL((fGroupMinimum.x + fGroupMaximum.x), f2vt(0.5f));
	fGroupCentre.y = VERTTYPEMUL((fGroupMinimum.y + fGroupMaximum.y), f2vt(0.5f));
	fGroupCentre.z = VERTTYPEMUL((fGroupMinimum.z + fGroupMaximum.z), f2vt(0.5f));

	/* No problem occured */
	return true;
}

//#pragma optimize( "", on )


void C3DSScene::SetFrame(const VERTTYPE fFrameNo)
{
	fFrame = fFrameNo;

	if (fFrame > f2vt(nNumFrames-1))
		fFrame = f2vt(0.0f);
}

/*!***************************************************************************
 @Function			Destroy
 @Description		De-allocate everything
*****************************************************************************/
void C3DSScene::Destroy()
{
	int i;

	/* Only release sub-pointers if main pointer was allocated! */
	if(pNode)
	{
		/* Loop through all frames */
		for(i = 0; i < nNumNodes; ++i)
		{
			/* Release memory taken up by each mesh animation structure in each frame */
			delete(pNode[i].pScaling);
			delete(pNode[i].pRotation);
			delete(pNode[i].pPosition);
		}
	}

	/* Release memory taken up by each node */
	delete(pNode);

    /* Only release sub-pointers if main pointer was allocated! */
	if(pMesh)
	{
		/* Loop through all meshes */
		for(i = 0; i < nNumMesh; ++i)
		{
			/* free all mesh data */
			delete(pMesh[i].pVertex);
			delete(pMesh[i].pFaces);
			delete(pMesh[i].pUV);
			delete(pMesh[i].pNormals);

			/* Reset number of vertices and faces */
			pMesh[i].nNumVertex= 0;
			pMesh[i].nNumFaces	= 0;
		}
	}

	/* Release memory taken up by meshes structures */
	delete(pMesh);

	/* Release memory taken up by cameras structures */
	delete(pCamera);

	/* Release memory taken up by lights structures */
	delete(pLight);

	/* Release memory taken up by materials structures */
	delete(pMaterial);

	/* Reset counters */
	nNumMesh		=  0;
	nNumMaterial	=  0;
	nNumCamera		=  0;
	nNumLight		=  0;
	nNumFrames		=  0;
}

/*!***************************************************************************
 @Function			GetCamera
 @Output			vFrom			Camera position at this frame
 @Output			vTo				Camera target at this frame
 @Input				nIdx				Number of the camera
 @Return			true or false
 @Description		Return position and target required to animate this camera at the
					specified frame. Frame number is a floating point number as (linear)
					interpolation will be used to compute the required matrix at
					this frame.
*****************************************************************************/
bool C3DSScene::GetCamera(VECTOR3	&vFrom, VECTOR3 &vTo, const unsigned int nIdx) const
{
	int			j;
	int			nKey1, nKey2;
	VERTTYPE	t;

	if(nIdx >= (unsigned int) nNumCamera)
		return false;

	S3DSNode *pCameraNode = 0;
	S3DSNode *pTargetNode = 0;

	if(pCamera[nIdx].NodeID != -1)
		pCameraNode = &pNode[pCamera[nIdx].NodeID];
	else
	{
		vFrom.x = pCamera[nIdx].fPosition.x;
		vFrom.y = pCamera[nIdx].fPosition.y;
		vFrom.z = pCamera[nIdx].fPosition.z;
	}

	if(pCamera[nIdx].TargetNodeID != -1)
		pTargetNode = &pNode[pCamera[nIdx].TargetNodeID];
	else
	{
		vTo.x = pCamera[nIdx].fTarget.x;
		vTo.y = pCamera[nIdx].fTarget.y;
		vTo.z = pCamera[nIdx].fTarget.z;
	}


	if(pCameraNode)
	{
		/* When there is no animation we just copy the first key */
		if(pCameraNode->PositionKeys <= 1)
		{
			vFrom.x = pCameraNode->pPosition[0].p.x;
			vFrom.y = pCameraNode->pPosition[0].p.y;
			vFrom.z = pCameraNode->pPosition[0].p.z;
		}
		else
		{
			/* Otherwise, go through all keys until we find the ones that contain fFrameNumber */
			for(j = 0; j < pCameraNode->PositionKeys - 1; ++j)
			{
				nKey1 = pCameraNode->pPosition[j].FrameNumber;
				nKey2 = pCameraNode->pPosition[j+1].FrameNumber;

				if(nKey1 <= (int) vt2f(fFrame) && nKey2 >= (int) vt2f(fFrame))
				{
					/* We are into, so interpolate the two positions */

					/* Compute t, the time corresponding to the frame we want to interpolate */
					t = VERTTYPEDIV((fFrame - f2vt(nKey1)) , f2vt(nKey2 - nKey1));

					/* Get interpolated position */
					MatrixVec3Lerp(vFrom, pCameraNode->pPosition[j].p, pCameraNode->pPosition[j+1].p, t);
				} // j loop
			}
		}
	}


    if(pTargetNode && pTargetNode->PositionKeys <= 1)
	{
		vTo.x = pTargetNode->pPosition[0].p.x;
		vTo.y = pTargetNode->pPosition[0].p.y;
		vTo.z = pTargetNode->pPosition[0].p.z;
		pTargetNode = 0;
	}

	if(pTargetNode)
	{
		/* When there is no animation we just copy the first key */
		if(pTargetNode->PositionKeys <= 1)
		{
			vTo.x = pTargetNode->pPosition[0].p.x;
			vTo.y = pTargetNode->pPosition[0].p.y;
			vTo.z = pTargetNode->pPosition[0].p.z;
		}
		else
		{
			/* Otherwise, go through all keys until we find the ones that contain fFrameNumber */
			for(j = 0; j < pTargetNode->PositionKeys - 1; ++j)
			{
				nKey1 = pTargetNode->pPosition[j].FrameNumber;
				nKey2 = pTargetNode->pPosition[j+1].FrameNumber;

				if(nKey1 <= (int) vt2f(fFrame) && nKey2 >= (int) vt2f(fFrame))
				{
					/* We are into, so interpolate the two positions */

					/* Compute t, the time corresponding to the frame we want to interpolate */
					t = VERTTYPEDIV((fFrame - f2vt(nKey1)), f2vt(nKey2 - nKey1));

					/* Get interpolated position */
					MatrixVec3Lerp(vTo, pTargetNode->pPosition[j].p, pTargetNode->pPosition[j+1].p, t);
				} // j loop
			}
		}
	}

	/* Return error */
	return true;
}

/*!***************************************************************************
 @Function			GetTransformationMatrix
 @Input				sMesh				a mesh
 @Output			mOut	Animation matrix at this frame
 @Return			true or false
 @Description		Return animation matrix required to animate this mesh at the
					specified frame.
					Frame number is a floating point number as (linear)
					interpolation will be used to compute the required matrix at
					this frame.
*****************************************************************************/
void C3DSScene::GetTransformationMatrix(MATRIX	&mOut, const S3DSMesh &sMesh)
{
	MATRIX	ObjectOffsetMatrix, WorldToLocalMatrix, TmpMatrix;

	if(sMesh.NodeID == -1)
	{
		MatrixIdentity(mOut);
		return;
	}

	/* Reset matrix to identity */
	MatrixIdentity(TmpMatrix);

	S3DSNode *p3DSNode = &pNode[sMesh.NodeID];

	/* Subtract pivot point from the matrix */
	TmpMatrix.f[12] -= p3DSNode->Pivot.x;
	TmpMatrix.f[13] -= p3DSNode->Pivot.y;
	TmpMatrix.f[14] -= p3DSNode->Pivot.z;

	/* Get final local-to-world matrix for this mesh at this frame */
 	ObjectOffsetMatrix = GetHierarchyMatrix((short)sMesh.NodeID, fFrame, TmpMatrix);

	/* Get final local-to-world matrix for this mesh at frame 0 */
	WorldToLocalMatrix = GetHierarchyMatrix((short)sMesh.NodeID, f2vt(0.0f), TmpMatrix);

	/* Invert y and z to get back to 3DS coordinates */
	TmpMatrix = WorldToLocalMatrix;

	WorldToLocalMatrix.f[ 1] = TmpMatrix.f[ 2];
	WorldToLocalMatrix.f[ 5] = TmpMatrix.f[ 6];
	WorldToLocalMatrix.f[ 9] = TmpMatrix.f[10];
	WorldToLocalMatrix.f[13] = TmpMatrix.f[14];
	WorldToLocalMatrix.f[ 2] = TmpMatrix.f[ 1];
	WorldToLocalMatrix.f[ 6] = TmpMatrix.f[ 5];
	WorldToLocalMatrix.f[10] = TmpMatrix.f[ 9];
	WorldToLocalMatrix.f[14] = TmpMatrix.f[13];

	/* Inverse matrix so that we can bring the model back to local coordinate */
	MatrixInverse(WorldToLocalMatrix, WorldToLocalMatrix);

	/* Concatenate this matrix with the local mesh matrix */
	MatrixMultiply(ObjectOffsetMatrix, WorldToLocalMatrix, ObjectOffsetMatrix);

	/* Re-invert y and z */
	mOut = ObjectOffsetMatrix;

	mOut.f[ 1] = ObjectOffsetMatrix.f[ 2];
	mOut.f[ 5] = ObjectOffsetMatrix.f[ 6];
	mOut.f[ 9] = ObjectOffsetMatrix.f[10];
	mOut.f[13] = ObjectOffsetMatrix.f[14];
	mOut.f[ 2] = ObjectOffsetMatrix.f[ 1];
	mOut.f[ 6] = ObjectOffsetMatrix.f[ 5];
	mOut.f[10] = ObjectOffsetMatrix.f[ 9];
	mOut.f[14] = ObjectOffsetMatrix.f[13];
}


/*!***************************************************************************
 @Function			DisplayInfo
 @Description		Display scene data into debug output
*****************************************************************************/
void C3DSScene::DisplayInfo()
{
	int		i;
	char	pszTmp[1024];

	/* Display total number of vertices and faces */
	sprintf(pszTmp, "Meshes : %d\nFaces : %d\n", nTotalVertices, nTotalFaces);
	_RPT0(_CRT_WARN,  pszTmp);

	/* Display centre and extremas */
	sprintf(pszTmp, "Meshes : %d\nGroup centre : (%f, %f, %f)\nMinimum : (%f, %f, %f)\nMaximum : (%f, %f, %f)\n",
		nNumMesh, vt2f(fGroupCentre.x), vt2f(fGroupCentre.y), vt2f(fGroupCentre.z),
		vt2f(fGroupMinimum.x), vt2f(fGroupMinimum.y), vt2f(fGroupMinimum.z),
		vt2f(fGroupMaximum.x), vt2f(fGroupMaximum.y), vt2f(fGroupMaximum.z));
	_RPT0(_CRT_WARN,  pszTmp);

	/* Display total number of meshes, nodes and frames */
	sprintf(pszTmp, "Meshes : %d\nNodes : %d\nFrames : %d\n", nNumMesh, nNumNodes, nNumFrames);
	_RPT0(_CRT_WARN,  pszTmp);

	/* Display light information */
	if(nNumLight)
	{
		for(i = 0; i < nNumLight; ++i)
		{
			sprintf(pszTmp, "Light %d : Position = (%.3f, %.3f, %.3f)\n        Colour = %.3f %.3f %.3f\n", i,
					vt2f(pLight[i].fPosition.x), vt2f(pLight[i].fPosition.y), vt2f(pLight[i].fPosition.z),
					vt2f(pLight[i].fColour[0]), vt2f(pLight[i].fColour[1]), vt2f(pLight[i].fColour[2]));
			_RPT0(_CRT_WARN,  pszTmp);
		}
	}
	else
	{
		_RPT0(_CRT_WARN,  "No light defined\n");
	}

	/* Display camera information */
	if(nNumCamera)
	{
		for(i = 0; i < nNumCamera; ++i)
		{
			sprintf(pszTmp, "Camera %d : Position = (%.3f, %.3f, %.3f)\n         Target = (%.3f %.3f %.3f)\n         Roll = %.3f  Focal length = %.3f\n", i,
					vt2f(pCamera[i].fPosition.x), vt2f(pCamera[i].fPosition.y), vt2f(pCamera[i].fPosition.z),
					vt2f(pCamera[i].fTarget.x), vt2f(pCamera[i].fTarget.y), vt2f(pCamera[i].fTarget.z),
					vt2f(pCamera[i].fRoll), vt2f(pCamera[i].fFocalLength));
			_RPT0(_CRT_WARN,  pszTmp);
		}
	}
	else
	{
		_RPT0(_CRT_WARN,  "No camera defined\n");
	}

	/* Display each mesh */
	if(nNumMesh)
	{
		for(i = 0; i < nNumMesh; ++i)
		{
			sprintf(pszTmp, "* Mesh %d: %s\n", i, pMesh[i].pszName);
			_RPT0(_CRT_WARN,  pszTmp);
			sprintf(pszTmp, "Material : %s (Number %d)\n", pMesh[i].pszMaterial, pMesh[i].nIdxMaterial);
			_RPT0(_CRT_WARN,  pszTmp);
			sprintf(pszTmp, "Mesh minimum : (%f, %f, %f)\n", vt2f(pMesh[i].fMinimum.x), vt2f(pMesh[i].fMinimum.y), vt2f(pMesh[i].fMinimum.z));
			_RPT0(_CRT_WARN,  pszTmp);
			sprintf(pszTmp, "Mesh maximum : (%f, %f, %f)\n", vt2f(pMesh[i].fMaximum.x), vt2f(pMesh[i].fMaximum.y), vt2f(pMesh[i].fMaximum.z));
			_RPT0(_CRT_WARN,  pszTmp);
			sprintf(pszTmp, "Mesh centre : (%f, %f, %f)\n", vt2f(pMesh[i].fCentre.x), vt2f(pMesh[i].fCentre.y), vt2f(pMesh[i].fCentre.z));
			_RPT0(_CRT_WARN,  pszTmp);
			sprintf(pszTmp, "Vertices : %d  Faces : %d\n", pMesh[i].nNumVertex, pMesh[i].nNumFaces);
			_RPT0(_CRT_WARN,  pszTmp);
			sprintf(pszTmp, "Pointers : Vertex = 0x%p   Faces = 0x%p   Normals = 0x%p   UV = 0x%p\n",
				pMesh[i].pVertex, pMesh[i].pFaces, pMesh[i].pNormals, pMesh[i].pUV);
			_RPT0(_CRT_WARN,  pszTmp);
		}
	}
	else
	{
		_RPT0(_CRT_WARN,  "No meshes defined\n");
	}

	/* Display each material */
	if(nNumMaterial)
	{
		for(i = 0; i < nNumMaterial; ++i)
		{
			sprintf(pszTmp, "Material %d:\n", i);
			_RPT0(_CRT_WARN,  pszTmp);

			sprintf(pszTmp, "Name : %s\nFile : %s\nOpaq : %s\n",
				pMaterial[i].pszMatName, pMaterial[i].pszMatFile, pMaterial[i].pszMatOpaq);
			_RPT0(_CRT_WARN,  pszTmp);

			sprintf(pszTmp, "Opacity : %d\n", pMaterial[i].nMatOpacity);
			_RPT0(_CRT_WARN,  pszTmp);

			sprintf(pszTmp, "Ambient : %f %f %f\nDiffuse : %f %f %f\nSpecular : %f %f %f\n",
				vt2f(pMaterial[i].fMatAmbient[0]) , vt2f(pMaterial[i].fMatAmbient[1]) , vt2f(pMaterial[i].fMatAmbient[2]),
				vt2f(pMaterial[i].fMatDiffuse[0]) , vt2f(pMaterial[i].fMatDiffuse[1]) , vt2f(pMaterial[i].fMatDiffuse[2]),
				vt2f(pMaterial[i].fMatSpecular[0]), vt2f(pMaterial[i].fMatSpecular[1]), vt2f(pMaterial[i].fMatSpecular[2]));
			_RPT0(_CRT_WARN,  pszTmp);

			sprintf(pszTmp, "Shininess : %f  Transparency : %f\n",
				vt2f(pMaterial[i].fMatShininess), vt2f(pMaterial[i].fMatTransparency));
			_RPT0(_CRT_WARN,  pszTmp);

			sprintf(pszTmp, "Shading : %d\n", pMaterial[i].sMatShading);
			_RPT0(_CRT_WARN,  pszTmp);
		}
	}
	else
	{
		_RPT0(_CRT_WARN,  "No materials defined\n");
	}
}


/*!***************************************************************************
 @Function			Scale
 @Input				Scale				Scale to apply
 @Description		Scale a model's vertices with a uniform scaling value
*****************************************************************************/
void C3DSScene::Scale(const VERTTYPE fScale)
{
	int i, j;

	/* Loop through all meshes */
	for (i = 0; i < nNumMesh; ++i)
	{
		/* Loop through all vertices in mesh */
		for (j = 0; j< pMesh[i].nNumVertex * 3; ++j)
		{
			/* Scale vertex */
			*(pMesh[i].pVertex + j) *= fScale;
		}
	}
}

/*!***************************************************************************
 @Function			GetHierarchyMatrix
 @Modified			nNode
 @Modified			fFrameNumber
 @Modified			CurrentMatrix
 @Return			Concatenated matrix
 @Description		Return total local-to-world matrix for this mesh
*****************************************************************************/
MATRIX C3DSScene::GetHierarchyMatrix(short nNode, VERTTYPE fFrameNumber, MATRIX CurrentMatrix)
{
	MATRIX		Matrix, PositionMatrix, RotationMatrix, ScalingMatrix;
	VECTOR3		Position, Scaling;
	int				i;
	int				nPreviousKey, nNextKey;
	VERTTYPE		t;

	/* Reset matrices */
	MatrixIdentity(PositionMatrix);
	MatrixIdentity(RotationMatrix);
	MatrixIdentity(ScalingMatrix);
	MatrixIdentity(Matrix);

	/*********************
	** Process position **
	*********************/
	/* Find previous key containing position information */
	nPreviousKey = -1;

	for (i = pNode[nNode].PositionKeys - 1; i >= 0; --i)
	{
		/* Is the first frame number below the one required in this key ? */
		if (pNode[nNode].pPosition[i].FrameNumber <= (int) vt2f(fFrameNumber))
		{
			/* Yes, store it and exit */
			nPreviousKey = i;
			break;
		}
	}

	/* Find next key containing position information */
	nNextKey = -1;
	for(i = 0; i < pNode[nNode].PositionKeys; ++i)
	{
		/* Is the first frame number above the one required in this key ? */
		if(pNode[nNode].pPosition[i].FrameNumber >= (int) vt2f(fFrameNumber) + 1)
		{
			/* Yes, store it and exit */
			nNextKey = i;
			break;
		}
	}

	/* If no previous key was found set position to the first valid key position (i.e next key) */
	if(nPreviousKey == -1)
	{
		Position = pNode[nNode].pPosition[nNextKey].p;
	}
	else
	{
		/* If no next key was found, set position to previous valid key position (i.e previous key) */
		if (nNextKey == -1)
		{
			Position = pNode[nNode].pPosition[nPreviousKey].p;
		}
		else
		{
			/* Interpolate between these two frames */

			/* Compute t, the time corresponding to the frame we want to interpolate */
			int nPrevFrame = pNode[nNode].pPosition[nPreviousKey].FrameNumber;
			int nNextFrame = pNode[nNode].pPosition[nNextKey].FrameNumber;

			t = VERTTYPEDIV((fFrameNumber - f2vt(nPrevFrame)), f2vt(nNextFrame - nPrevFrame));

			/* Get interpolated position */
			MatrixVec3Lerp(Position, pNode[nNode].pPosition[nPreviousKey].p, pNode[nNode].pPosition[nNextKey].p, t);
		}
	}

	/* Perform translation */
	MatrixTranslation(PositionMatrix, Position.x, Position.y, Position.z);

	/* Concatenate matrices */
	MatrixMultiply(Matrix, PositionMatrix, Matrix);


	/*********************
	** Process rotation **
	*********************/
	/* Get absolute rotation at this frame */
	GetAbsoluteRotation(&RotationMatrix, fFrameNumber, &pNode[nNode]);

	/* Concatenate matrices */
	MatrixMultiply(Matrix, RotationMatrix, Matrix);


	/********************
	** Process scaling **
	********************/
	/* Find previous key containing scaling information */
	nPreviousKey = -1;

	for(i = pNode[nNode].ScalingKeys- 1; i >= 0; --i)
	{
		/* Is the first frame number below the one required in this key ? */
		if(pNode[nNode].pScaling[i].FrameNumber <= (int) vt2f(fFrameNumber))
		{
			/* Yes, store it and exit */
			nPreviousKey = i;
			break;
		}
	}

	/* Find next key containing scaling information */
	nNextKey = -1;

	for (i=0; i < pNode[nNode].ScalingKeys; ++i)
	{
		/* Is the first frame number above the one required in this key ? */
		if(pNode[nNode].pScaling[i].FrameNumber >= (int) vt2f(fFrameNumber) + 1)
		{
			/* Yes, store it and exit */
			nNextKey = i;
			break;
		}
	}

	/* If no previous key was found set scaling to the first valid key scaling (i.e next key) */
	if(nPreviousKey == -1)
	{
		Scaling = pNode[nNode].pScaling[nNextKey].s;
	}
	else
	{
		/* If no next key was found, set scaling to previous valid key scaling (i.e previous key) */
		if(nNextKey == -1)
		{
			Scaling = pNode[nNode].pScaling[nPreviousKey].s;
		}
		else
		{
			/* Interpolate between these two frames */

			/* Compute t, the time corresponding to the frame we want to interpolate */
			int nPrevFrame = pNode[nNode].pScaling[nPreviousKey].FrameNumber;
			int nNextFrame = pNode[nNode].pScaling[nNextKey].FrameNumber;

			t = VERTTYPEDIV((fFrame - f2vt(nPrevFrame)), f2vt(nNextFrame - nPrevFrame));

			/* Get interpolated scaling */
			MatrixVec3Lerp(Scaling, pNode[nNode].pScaling[nPreviousKey].s, pNode[nNode].pScaling[nNextKey].s, t);
		}
	}

	/* Perform scaling */
	ScalingMatrix.f[0] = Scaling.x;
	ScalingMatrix.f[5] = Scaling.y;
	ScalingMatrix.f[10] = Scaling.z;

	/* Concatenate matrices */
	MatrixMultiply(Matrix, ScalingMatrix, Matrix);


	/***********************************
	** Concatenate with parent matrix **
	***********************************/
	MatrixMultiply(CurrentMatrix, CurrentMatrix, Matrix);


	/*************************
	** Recurse in hierarchy **
	*************************/
	if(pNode[nNode].ParentIndex != -1 && pNode[nNode].ParentIndex != nNode)
	{
		CurrentMatrix = GetHierarchyMatrix(pNode[nNode].ParentIndex, fFrameNumber, CurrentMatrix);
	}

	/* Return matrix */
	return CurrentMatrix;
}


/*!***************************************************************************
 @Function			GetAbsoluteRotation
 @Input				nFrameNumber
 @Input				pNode
 @Return			Quaternion corresponding to rotation at specified frame
 @Description		Return quaternion corresponding to rotation at specified
					frame. Interpolation might be used if no key contains the
					specified frame number.
*****************************************************************************/
void C3DSScene::GetAbsoluteRotation(MATRIX * const pmRot, VERTTYPE fFrameNumber, S3DSNode *pNode)
{
	QUATERNION	TmpQuaternion, RotationQuaternion, Q1, Q2;
	int			i, nPreviousKey, nNextKey;
	VERTTYPE		t;


	/* Find previous key containing rotation information */
	nPreviousKey = -1;

	for(i = pNode->RotationKeys - 1; i >= 0; --i)
	{
		/* Is the first frame number below the one required in this key ? */
		if(pNode->pRotation[i].FrameNumber <= (int) vt2f(fFrameNumber))
		{
			/* Yes, store it and exit */
			nPreviousKey = i;
			break;
		}
	}

	/* Find next key containing rotation information */
	nNextKey = -1;

	for(i = 0 ; i < pNode->RotationKeys; ++i)
	{
		/* Is the first frame number above the one required in this key ? */
		if (pNode->pRotation[i].FrameNumber >= (int) vt2f(fFrameNumber) + 1)
		{
			/* Yes, store it and exit */
			nNextKey = i;
			break;
		}
	}

	/* If no previous key was found set rotation to the first valid key rotation (i.e next key) */
	if(nPreviousKey == -1)
	{
		/* Create quaternion from vector and angle */
		MatrixQuaternionRotationAxis(RotationQuaternion, pNode->pRotation[nNextKey].r, pNode->pRotation[nNextKey].Angle);
		MatrixRotationQuaternion(*pmRot, RotationQuaternion);
		MatrixTranspose(*pmRot, *pmRot);
		return;
	}
	else
	{
		/* Compute absolute quaternion for previous key */

		/* Initialise quaternion */
		MatrixQuaternionIdentity(Q1);

		for(i = 0; i <= nPreviousKey; ++i)
		{
			/* Create quaternion from vector and angle */
			MatrixQuaternionRotationAxis(TmpQuaternion, pNode->pRotation[i].r, pNode->pRotation[i].Angle);

			/* Multiply quaternion with previous one */
			MatrixQuaternionMultiply(Q1, Q1, TmpQuaternion);
		}

		/* If no next key was found, set rotation to previous valid key rotation (i.e previous key) */
		if(nNextKey == -1)
		{
			/* Convert quaternion to matrix & return result */
			MatrixRotationQuaternion(*pmRot, Q1);
			MatrixTranspose(*pmRot, *pmRot);
			return;
		}
	}

	/* Compute absolute quaternion for next key */

	/* Create quaternion from vector and angle */
	MatrixQuaternionRotationAxis(TmpQuaternion, pNode->pRotation[nNextKey].r, pNode->pRotation[nNextKey].Angle);

	/* Multiply quaternion with previous one */
	MatrixQuaternionMultiply(Q2, Q1, TmpQuaternion);


	/* We now have our two quaternions Q1 and Q2, let's interpolate */

	int nPrevFrame = pNode->pRotation[nPreviousKey].FrameNumber;
	int nNextFrame = pNode->pRotation[nNextKey].FrameNumber;

	/* Compute t, the time corresponding to the frame we want to interpolate */
	t = VERTTYPEDIV((fFrameNumber - f2vt(nPrevFrame)),f2vt(nNextFrame - nPrevFrame));

	/* Using linear interpolation, find quaternion at time t between Q1 and Q2 */
	MatrixQuaternionSlerp(RotationQuaternion, Q1, Q2, t);

	/* Convert quaternion to matrix & return result */
	MatrixRotationQuaternion(*pmRot, RotationQuaternion);
	MatrixTranspose(*pmRot, *pmRot);
}

/*!***************************************************************************
 @Function			Normal
 @Input				pV1
 @Input				pV2
 @Input				pV3
 @Output			NormalVect
 @Description		Compute the normal to the triangle defined by the vertices V1,
					V2 and V3.
*****************************************************************************/
VECTOR3 C3DSScene::Normal(VERTTYPE *pV1, VERTTYPE *pV2, VERTTYPE *pV3)
{
	/*
		The calculation of the normal will be done in floating point,
		doesn't matter if we're using fixed point.
	*/
	VECTOR3 fNormal;
	VECTOR3 fV1, fV2, fV3;
	VECTOR3 Vect1, Vect2;

	fV1.x = vt2f(pV1[0]); fV1.y = vt2f(pV1[1]); fV1.z = vt2f(pV1[2]);
	fV2.x = vt2f(pV2[0]); fV2.y = vt2f(pV2[1]); fV2.z = vt2f(pV2[2]);
	fV3.x = vt2f(pV3[0]); fV3.y = vt2f(pV3[1]); fV3.z = vt2f(pV3[2]);

	float PMod;

    /* Compute triangle vectors */
	Vect1.x = fV1.x-fV2.x;   Vect1.y = fV1.y-fV2.y;   Vect1.z = fV1.z-fV2.z;
    Vect2.x = fV1.x-fV3.x;   Vect2.y = fV1.y-fV3.y;   Vect2.z = fV1.z-fV3.z;

	/* Find cross-product vector of these two vectors */
	fNormal.x = (Vect1.y * Vect2.z) - (Vect1.z * Vect2.y);
	fNormal.y = (Vect1.z * Vect2.x) - (Vect1.x * Vect2.z);
	fNormal.z = (Vect1.x * Vect2.y) - (Vect1.y * Vect2.x);

	/* Compute length of the resulting vector */
    PMod = (float)sqrt(fNormal.x*fNormal.x+fNormal.y*fNormal.y+fNormal.z*fNormal.z);

	/* This is to avoid a division by zero */
    if (PMod < 1e-10f)
		PMod = 1e-10f;

	PMod = 1.0f / PMod;

	/* Normalize normal vector */
    fNormal.x *= PMod;
	fNormal.y *= PMod;
	fNormal.z *= PMod;

	return fNormal;
}


/*!***************************************************************************
 @Function			CalculateNormals
 @Modified			pObject
 @Description		Compute vertex normals of submitted vertices array.
*****************************************************************************/
void C3DSScene::CalculateNormals(int nNumVertex, VERTTYPE *pVertex,
							 int nNumFaces, unsigned short *pFaces,
							 VERTTYPE *pNormals)
{
	unsigned short	P1, P2, P3;
	VERTTYPE		fMod, *pfVN;
	int				nIdx;
	int				j, k;

	VECTOR3 fNormal;

	// Parameter checking
	if (!pVertex || !pFaces || !pNormals)
	{
		_RPT0(_CRT_WARN,  "CalculateNormals : Bad parameters\n");
		return;
	}

	// Use the actual output array for summing face-normal contributions
	pfVN = pNormals;

	// Zero normals array
	memset(pfVN, 0, nNumVertex * 3 * sizeof(VERTTYPE));

	// Sum the components of each face's normal to a vector normal
	for (j=0; j < 3 * nNumFaces; j += 3)
	{
		// Get three points defining a triangle
		P1 = pFaces[j + 0];
		P2 = pFaces[j + 1];
		P3 = pFaces[j + 2];

		// Calculate face normal in pfN
		fNormal = Normal(&pVertex[3*P1], &pVertex[3*P2], &pVertex[3*P3]);

		// Add the normal of this triangle to each vertex
		for (k=0; k<3; k++)
		{
			nIdx = pFaces[j + k];
			pfVN[nIdx * 3 + 0] += f2vt(fNormal.x);
			pfVN[nIdx * 3 + 1] += f2vt(fNormal.y);
			pfVN[nIdx * 3 + 2] += f2vt(fNormal.z);
		}
	}

	VERTTYPE fSq[3];
	// Normalise each vector normal and set in mesh
	for (j = 0; j < 3 * nNumVertex; j += 3)
	{

		fSq[0] = VERTTYPEMUL(pfVN[j + 0], pfVN[j + 0]);
		fSq[1] = VERTTYPEMUL(pfVN[j + 1], pfVN[j + 1]);
		fSq[2] = VERTTYPEMUL(pfVN[j + 2], pfVN[j + 2]);

		fMod = (VERTTYPE) f2vt(sqrt(vt2f(fSq[0] + fSq[1] + fSq[2])));

		// Zero length normal? Either point down an axis or leave...
		if(fMod == f2vt(0.0f))
			continue;

		fMod = VERTTYPEDIV(f2vt(1.0f), fMod);

		pNormals[j + 0] = VERTTYPEMUL(pNormals[j + 0], fMod);
		pNormals[j + 1] = VERTTYPEMUL(pNormals[j + 1], fMod);
		pNormals[j + 2] = VERTTYPEMUL(pNormals[j + 2], fMod);
	}
}

/*****************************************************************************
 End of file (Model3DS.cpp)
*****************************************************************************/
