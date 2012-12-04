
#ifndef BT_COLLIDABLE_H
#define BT_COLLIDABLE_H

struct btCollidable
{
	int m_shapeType;
	int m_shapeIndex;
};

struct btGpuChildShape
{
	float	m_childPosition[4];
	float	m_childOrientation[4];
	int m_shapeIndex;
};

#endif //BT_COLLIDABLE_H
