MSTRINGIFY(




__kernel void 
  broadphaseKernel( const int startOffset, const int numNodes, __global float4 *g_vertexBuffer,
		   __global float4 *linVel,
		   __global float4 *pAngVel,
		   __global float* pBodyTimes)
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