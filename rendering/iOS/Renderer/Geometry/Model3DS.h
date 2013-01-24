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
#ifndef _MODEL3DS_H_
#define _MODEL3DS_H_

#include "Mathematics.h"
#include "MemoryManager.h"

/*!***************************************************************************
 Mesh structure
*****************************************************************************/
struct S3DSMesh {
	char		pszName[20];		/*!< Name of mesh */
    char		pszMaterial[20];	/*!< Name of material used for this mesh */
	int			nIdxMaterial;	/*!< Index of material used in this mesh */
	VECTOR3	fCentre;			/*!< Mesh centre */
	VECTOR3	fMinimum;		/*!< Bounding box's lower front corner */
	VECTOR3	fMaximum;		/*!< Bounding box's upper back corner */
	int			nNumVertex;			/*!< Number of vertices in the mesh */
	int			nNumFaces;			/*!< Number of triangles in the mesh */
	VERTTYPE		*pVertex;			/*!< List of vertices (x0, y0, z0, x1, y1, z1, x2, etc...) */
	unsigned short	*pFaces;	/*!< List of triangles indices */
	VERTTYPE		*pNormals;			/*!< List of vertex normals (Nx0, Ny0, Nz0, Nx1, Ny1, Nz1, Nx2, etc...) */
	VERTTYPE		*pUV;				/*!< List of UV coordinate (u0, v0, u1, v1, u2, etc...) */
	short		NodeID;
};

/*!***************************************************************************
 Material structure
*****************************************************************************/
struct S3DSMaterial {
       char		pszMatName[256];	/*!< Material name */
       char		pszMatFile[256];	/*!< Material file name (used if textured) */
	   char		pszMatOpaq[256];	/*!< ? */
	   int		nMatOpacity;		/*!< Material opacity (used with vertex alpha ?) */
       VERTTYPE	fMatAmbient[3];		/*!< Ambient RGB value for material */
       VERTTYPE	fMatDiffuse[3];		/*!< Diffuse RGB value for material */
       VERTTYPE	fMatSpecular[3];	/*!< Specular RGB value for material */
       VERTTYPE	fMatShininess;		/*!< Material shininess */
       VERTTYPE	fMatTransparency;	/*!< Material transparency */
       short	sMatShading;		/*!< Shading mode used with this material */
};

/*!***************************************************************************
 Light structure
*****************************************************************************/
struct S3DSLight {
		VECTOR3	fPosition;		/*!< Light position in World coordinates */
		VERTTYPE	fColour[3];		/*!< Light colour (0.0f -> 1.0f for each channel) */
};

/*!***************************************************************************
 Camera structure
*****************************************************************************/
struct S3DSCamera {
		char	pszName[20];		/*!< Name of camera */
		VECTOR3	fPosition;		/*!< Camera position */
		VECTOR3	fTarget;			/*!< Camera looking point */
		VERTTYPE	fRoll;				/*!< Camera roll value */
		VERTTYPE	fFocalLength;		/*!< Camera focal length, in millimeters */
		VERTTYPE	fFOV;				/*!< Field of view */
		short	NodeID;
		short   TargetNodeID;
};

/*!***************************************************************************
 Keyframe position structure
*****************************************************************************/
struct POSITIONKEY {

	int			FrameNumber;
	VERTTYPE		fTension;
	VERTTYPE		fContinuity;
	VERTTYPE		fBias;
	VECTOR3	p;
};

/*!***************************************************************************
 Keyframe rotation structure
*****************************************************************************/
struct ROTATIONKEY {

	int			FrameNumber;
	VERTTYPE	fTension;
	VERTTYPE	fContinuity;
	VERTTYPE	fBias;
	VERTTYPE	Angle;
	VECTOR3	r;
};

/*!***************************************************************************
 Keyframe scaling structure
*****************************************************************************/
struct SCALINGKEY {

	int			FrameNumber;
	VERTTYPE	fTension;
	VERTTYPE	fContinuity;
	VERTTYPE	fBias;
	VECTOR3	s;
};

/*!***************************************************************************
 Node structure
*****************************************************************************/
struct S3DSNode {
	short		ParentIndex;	/*!< Parent node Id */
	VECTOR3	Pivot;		/*!< Pivot for this mesh */

	/* Position */
	int			PositionKeys;
	POSITIONKEY	*pPosition;

	/* Rotation */
	int			RotationKeys;
	ROTATIONKEY	*pRotation;

	/* Scaling */
	int			ScalingKeys;
	SCALINGKEY	*pScaling;
};


/*!***************************************************************************
 Object structure
*****************************************************************************/
struct S3DSScene {

	/* Information related to all meshes */
	int				nTotalVertices;		/*!< Total number of vertices in object */
	int				nTotalFaces;		/*!< Total number of faces in object */
	VECTOR3		fGroupCentre;	/*!< Centre of object */
	VECTOR3		fGroupMinimum;	/*!< Bounding box's lower front corner */
	VECTOR3		fGroupMaximum;	/*!< Bounding box's upper back corner */

	/* Meshes defined in the .3DS file */
	int				nNumMesh;			/*!< Number of meshes composing the object */
	S3DSMesh		*pMesh;				/*!< List of meshes in object */

	/* Materials defined in the .3DS file */
	int				nNumMaterial;		/*!< Number of materials used with object */
	S3DSMaterial	*pMaterial;			/*!< List of materials used with object */

	/* Lights defined in the .3DS file */
	int				nNumLight;			/*!< Number of lights */
	S3DSLight		*pLight;			/*!< List of lights */

	/* Cameras defined in the .3DS file */
	int				nNumCamera;		/*!< Number of cameras */
	S3DSCamera		*pCamera;			/*!< List of cameras */

	/* This is used for animation only. Ignore if animation is not to be used */
	int				nNumFrames;			/*!< Number of frames */
	int				nNumNodes;			/*!< Number of nodes */
	S3DSNode		*pNode;				/*!< List of nodes */

	VERTTYPE		fFrame;			/*!< Frame number */
};

/***************
** Prototypes **
***************/

class C3DSScene : public S3DSScene
{
	public:
	/*!***************************************************************************
	@Function			ReadFromFile
	@Input				pszFileName			Name of the 3DS file to load
	@Return				true or false
	@Description		Read a 3DS file. The 3DS file can either be a resource or file.
						If stored in a resource, the resource identifier must be "TDS".
	*****************************************************************************/
	bool ReadFromFile(const char * const pszFileName);

	/*!***************************************************************************
	@Function			Destroy
	@Return				true or false
	@Description		Destroy the scene
	*****************************************************************************/
	void Destroy();
	/*!***************************************************************************
	@Function			DisplayInfo
	@Description		Display model data into debug output
	*****************************************************************************/
	void DisplayInfo();
	/*!***************************************************************************
	@Function			Scale
	@Input				fScale				Scale to apply
	@Description		Scale a model's vertices with a uniform scaling value
	*****************************************************************************/
	void Scale(const VERTTYPE fScale);

	/*!***************************************************************************
		@Function		SetFrame
		@Input			fFrameNo			Frame number
		@Description	Set the animation frame for which subsequent Get*() calls
						should return data.
	*****************************************************************************/
	void SetFrame(const VERTTYPE fFrameNo);

	/*!***************************************************************************
	@Function			GetTransformationMatrix
	@Input				node			3DS Mesh to return matrix for.
	@Output				mOut			Animation matrix at this frame
	@Description		Return animation matrix required to animate this mesh at the
						current frame. The frame number is a floating point number as
						(linear) interpolation will be used for getting the animation data.
	*****************************************************************************/
	void GetTransformationMatrix(MATRIX	&mOut, const S3DSMesh &node);
	/*!***************************************************************************
	@Function			GetCamera
	@Output				vFrom			Camera position at the current frame
	@Output				vTo				Camera target at the current frame
	@Input				nIdx				Number of the camera
	@Return				true or false
	@Description		Return position and target required to animate this camera at the
						current frame.
	*****************************************************************************/
	bool GetCamera(VECTOR3	&vFrom,VECTOR3 &vTo, const unsigned int nIdx) const;

protected:
	MATRIX GetHierarchyMatrix(short Node, VERTTYPE fFrameNumber, MATRIX CurrentMatrix);
	void GetAbsoluteRotation(MATRIX * const pmRot, VERTTYPE fFrameNumber, S3DSNode *pNode);
	VECTOR3 Normal(VERTTYPE *pV1, VERTTYPE *pV2, VERTTYPE *pV3);
	void CalculateNormals(int nNumVertex, VERTTYPE *pVertex, int nNumFaces, unsigned short *pFaces, VERTTYPE *pNormals);
	
//private:
//    DECLARE_HEAP;

};


#endif
