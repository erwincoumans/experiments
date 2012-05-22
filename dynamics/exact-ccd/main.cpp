#include <stdio.h>
#include <float.h>
//#pragma fenv_access (on)
#include <assert.h>

#include "rootparitycollisiontest.h"
#include "vec.h"

int main(int argc, char* argv[])
{
	assert((_controlfp(0,0)&_MCW_RC)==_RC_NEAR);
	bool result =false;

	{
		//test vertex-triangle
		Vec3d verts_old[4]={Vec3d(-2,0,1.0000001),Vec3d(0,0,1),Vec3d(-1,0,-1),Vec3d(1,0,-1)};
		Vec3d verts_new[4]={Vec3d(2.1,0,0.999999),Vec3d(0,0,1),Vec3d(-1,0,-1),Vec3d(1,0,-1)};

		bool is_edge_edge = false;

		rootparity::RootParityCollisionTest test(
			verts_old[0],verts_old[1],verts_old[2],verts_old[3],
			verts_new[0],verts_new[1],verts_new[2],verts_new[3],is_edge_edge);

		result = test.run_test();
	}

	{
		//test edge-edge
		Vec3d verts_old[4]={Vec3d(-1,0,0),Vec3d(1,0,0),Vec3d(0,1,-1),Vec3d(0,1,1)};
		Vec3d verts_new[4]={Vec3d(-1,0,0),Vec3d(1,0,0),Vec3d(0,0,-1),Vec3d(0,0,1)};

		bool is_edge_edge = true;

		rootparity::RootParityCollisionTest test(
			verts_old[0],verts_old[1],verts_old[2],verts_old[3],
			verts_new[0],verts_new[1],verts_new[2],verts_new[3],is_edge_edge);

		result = test.run_test();
	}

	return 0;
}