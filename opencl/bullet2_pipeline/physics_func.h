/*
Physics Effects Copyright(C) 2010 Sony Computer Entertainment Inc.
All rights reserved.

Physics Effects is open software; you can redistribute it and/or
modify it under the terms of the BSD License.

Physics Effects is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the BSD License for more details.

A copy of the BSD License is distributed with
Physics Effects under the filename: physics_effects_license.txt
*/

#ifndef __BULLET2_PHYSICS_FUNC_H__
#define __BULLET2_PHYSICS_FUNC_H__


//E Simulation
//J シミュレーション
bool physics_init();
void physics_release();
void physics_create_scene(int sceneId);
void physics_simulate();


//E Change parameters
//J パラメータの取得
int physics_get_num_rigidbodies();

int physics_get_num_contacts();

class btCollisionObject* physics_get_collision_object(int objectIndex);

#endif /* __BULLET2_PHYSICS_FUNC_H__ */
