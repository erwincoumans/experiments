/*
 GameKit .blend file reader for Oolong Engine
 Copyright (c) 2009 Erwin Coumans http://gamekit.googlecode.com
 
 This software is provided 'as-is', without any express or implied warranty.
 In no event will the authors be held liable for any damages arising from the use of this software.
 Permission is granted to anyone to use this software for any purpose, 
 including commercial applications, and to alter it and redistribute it freely, 
 subject to the following restrictions:
 
 1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
 2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
 3. This notice may not be removed or altered from any source distribution.
 */

#ifndef OOLONG_READ_BLEND_H
#define OOLONG_READ_BLEND_H

#include "BulletBlendReaderNew.h"
#include "LinearMath/btTransform.h"
#include "LinearMath/btAlignedObjectArray.h"
#include "LinearMath/btHashMap.h"

class btCollisionObject;

ATTRIBUTE_ALIGNED16(struct) GfxVertex
{
	BT_DECLARE_ALIGNED_ALLOCATOR();

	btVector3 m_position;
	btVector3 m_normal;
	float	m_uv[2];
};

ATTRIBUTE_ALIGNED16(struct) GfxObject
{
	BT_DECLARE_ALIGNED_ALLOCATOR();
	struct BasicTexture*	m_texture;
	
	int	m_numVerts;
	btAlignedObjectArray<unsigned short int>	m_indices;
	btAlignedObjectArray<GfxVertex>	m_vertices;
	
	btCollisionObject* m_colObj;
	
	GfxObject(GLuint vboId,btCollisionObject* colObj);
	btVector3 m_scaling;
	
	void render(int positionLoc, int textureLoc,int samplerLoc,int modelMatrix);
	
};

class OolongBulletBlendReader : public BulletBlendReaderNew
{
public:
	btAlignedObjectArray<GfxObject>	m_graphicsObjects;
	
	btHashMap<btHashString,BasicTexture*> m_textures;
		
	btTransform m_cameraTrans;
	
	BasicTexture*	m_notFoundTexture;
	
	BasicTexture* findTexture(const char* fileName);
	
	OolongBulletBlendReader(class btDynamicsWorld* destinationWorld);
	
	virtual ~OolongBulletBlendReader();
	
	///for each Blender Object, this method will be called to convert/retrieve data from the bObj
	virtual void*   createGraphicsObject(Blender::Object* tmpObject, class btCollisionObject* bulletObject);

	
	virtual	void	addCamera(Blender::Object* tmpObject);
	
	virtual	void	addLight(Blender::Object* tmpObject);
	
	virtual void	convertLogicBricks();
	
	virtual void	createParentChildHierarchy();
	
};


#endif //OOLONG_READ_BLEND_H


