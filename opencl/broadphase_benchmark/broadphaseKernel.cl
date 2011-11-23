MSTRINGIFY(


typedef struct 
{
	float			fx;
	float			fy;
	float			fz;
	unsigned int	uw;
} btAABBCL;

__kernel void 
  broadphaseGridKernel( const int startOffset, const int numNodes, __global float4 *g_vertexBuffer, __global btAABBCL* pAABB)
{
	int nodeID = get_global_id(0);
		
	if( nodeID < numNodes )
	{
		float4 position = g_vertexBuffer[nodeID + startOffset/4];
		//float4 orientation = g_vertexBuffer[nodeID + startOffset/4+numNodes];
		//float4 color = g_vertexBuffer[nodeID + startOffset/4+numNodes+numNodes];
		
		float4 green = (float4)(0.f,1.f,0.f,0.f);
		g_vertexBuffer[nodeID + startOffset/4+numNodes+numNodes] = green;
		
		pAABB[nodeID*2].fx = position.x-1.f;
		pAABB[nodeID*2].fy = position.y-1.f;
		pAABB[nodeID*2].fz = position.z-1.f;
		pAABB[nodeID*2].uw = nodeID;

		pAABB[nodeID*2+1].fx = position.x+1.f;
		pAABB[nodeID*2+1].fy = position.y+1.f;
		pAABB[nodeID*2+1].fz = position.z+1.f;
		pAABB[nodeID*2+1].uw = nodeID;		
	}
}


__kernel void 
  broadphaseColorKernel( const int startOffset, const int numNodes, __global float4 *g_vertexBuffer, __global int2* pOverlappingPairs, const int numOverlap)
{
	int nodeID = get_global_id(0);
	if( nodeID < numOverlap )
	{
		int2 pair = pOverlappingPairs[nodeID];
		float4 red = (float4)(1.f,0.f,0.f,0.f);
		
		g_vertexBuffer[pair.x + startOffset/4+numNodes+numNodes] = red;
		g_vertexBuffer[pair.y + startOffset/4+numNodes+numNodes] = red;
	}
}



__kernel void 
  broadphaseKernel( const int startOffset, const int numNodes, __global float4 *g_vertexBuffer)
{
	int nodeID = get_global_id(0);
	
//	float BT_GPU_ANGULAR_MOTION_THRESHOLD = (0.25f * 3.14159254);
	
	if( nodeID < numNodes )
	{
		float4 position = g_vertexBuffer[nodeID + startOffset/4];
		//float4 orientation = g_vertexBuffer[nodeID + startOffset/4+numNodes];
		float4 color = g_vertexBuffer[nodeID + startOffset/4+numNodes+numNodes];
		
		float4 red = (float4)(1.f,0.f,0.f,0.f);
		float4 green = (float4)(0.f,1.f,0.f,0.f);
		float4 blue = (float4)(0.f,0.f,1.f,0.f);
		float  overlap=0;
		int equal = 0;
		
		g_vertexBuffer[nodeID + startOffset/4+numNodes+numNodes] = green;
		
		for (int i=0;i<numNodes;i++)
		{
			if (i!=nodeID)
			{
				float4 otherPosition = g_vertexBuffer[i + startOffset/4];
				if ((otherPosition.x == position.x)&&
					(otherPosition.y == position.y)&&
					(otherPosition.z == position.z))
						equal=1;
				
				
				float distsqr = 
						((otherPosition.x - position.x)* (otherPosition.x - position.x))+
						((otherPosition.y - position.y)* (otherPosition.y - position.y))+
						((otherPosition.z - position.z)* (otherPosition.z - position.z));
				
				if (distsqr<7.f)
					overlap+=0.25f;
			}
		}
		
		
		if (equal)
		{
				g_vertexBuffer[nodeID + startOffset/4+numNodes+numNodes]=blue;
		} else
		{
			if (overlap>0.f)
				g_vertexBuffer[nodeID + startOffset/4+numNodes+numNodes]=red*overlap;
			else
				g_vertexBuffer[nodeID + startOffset/4+numNodes+numNodes]=green;
		}
	}
}

);