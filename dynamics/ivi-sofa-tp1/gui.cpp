#include <sofa/helper/system/gl.h>
#include <sofa/helper/system/glu.h>
#include <sofa/helper/system/glut.h>

#include <iostream>

#include "simulation.h"
#include "render.h"

// Methods from glut_main.cpp

extern int window_width, window_height;
extern void drawTextPos(float x, float y);
extern void drawTextAlign(float h, float v);
extern void drawTextFont(float size, bool serif = false);
extern float drawText(const char* str);
extern float drawTextF(const char* fmt, ...);

// Misc displayed info

extern double fps;
extern int simulation_cg_iter;
extern std::string device_name;

//// DATA ////

TCoord camera_lookat;
double camera_distance;
TCoord camera_direction;

int picked_particle = -1;
int fixed_particle = -1;
TCoord picked_origin;
TCoord picked_dir;

//// METHODS ////

void reset_camera()
{
    camera_lookat = simulation_center;
    camera_lookat[2] += simulation_size*0.1f;
    camera_distance = simulation_size*1.2f;
    camera_direction = TCoord(1.0f,0.25f,1.0f);
    camera_direction.normalize();
    camera_position = simulation_center + camera_direction * camera_distance;
}

void move_camera(int dx, int dy, int mode)
{
    TCoord vx = cross(camera_direction, TCoord(0,1,0));
    TCoord vy = cross(vx,camera_direction);
    vx.normalize();
    vy.normalize();
    switch (mode)
    {
    case 0: // rotate
        camera_direction += (vx*dx + vy*dy)*0.005;
        camera_direction.normalize();
        break;
    case 1: // translate
        camera_lookat += (vx*dx + vy*dy)*(camera_distance*0.001);
        break;
    case 2: // zoom
        camera_distance *= pow(0.99,dy);
        break;
    }
}

TCoord picked_position()
{
    // dot(origin + t * dir - particle, dir) = 0
    // dot(origin-particle,dir) + dot(t * dir, dir) = 0
    // t = -dot(origin-particle,dir)
    // pos = origin + dir * dot(particle-origin,dir)
    TCoord particle;
    if (picked_particle == -1) particle = simulation_center;
    else particle = fem_mesh->positions[picked_particle];
    return picked_origin + picked_dir * dot(particle-picked_origin, picked_dir);
}

int start_picking(int picking_mode, int x, int y, int bt, double obj_x, double obj_y, double obj_z)
{
    //picked_position = TCoord(obj_x, obj_y, obj_z);
    picked_particle = -1;
    if (!fem_mesh) return -1;
    if (fixed_particle != -1)
    {
        fem_mesh->removeFixedParticle(fixed_particle);
        fixed_particle = -1;
    }
    picked_origin = camera_position;
    picked_dir = TCoord(obj_x, obj_y, obj_z) - picked_origin;
    picked_dir.normalize();
    int index = -1;
    TReal dotmargin = (TReal)0.9995;
    TReal mindist = simulation_size*10;
    unsigned int nbp = fem_mesh->positions.size();
    const TCoord* pos = fem_mesh->positions.hostRead();
    for (unsigned int i=0;i<nbp;++i)
    {
        TCoord p = pos[i];
        p -= picked_origin;
        TReal pdist = p.norm();
        TReal pdot = dot(p,picked_dir);
        // we add to the distance to the particle 10 times the distance to the ray to prefer the closest particle of the same surface
        TReal dist = pdist + 10 * (picked_dir * pdot - p).norm();
        if (pdot / pdist > dotmargin && dist < mindist)
        {
            index = i;
            mindist = dist;
        }
    }
    if (index == -1) return -1;
    picked_particle = index;

    return update_picking(1+bt);

}

int move_picking(int picking_mode, int x, int y, int dx, int dy, double obj_x, double obj_y, double obj_z)
{
    picked_origin = camera_position;
    picked_dir = TCoord(obj_x, obj_y, obj_z) - picked_origin;
    picked_dir.normalize();
    if (!fem_mesh) return -1;
    return update_picking(picking_mode);
}

int update_picking(int picking_mode)
{
    if (!fem_mesh) return -1;
    if (picking_mode != -1 && picked_particle != -1)
    {
        switch(picking_mode)
        {
        case 3: // middle click
            if (fixed_particle == -1)
            {
                if (!fem_mesh->isFixedParticle(picked_particle))
                {
                    fem_mesh->addFixedParticle(picked_particle);
                    fixed_particle = picked_particle;
                }
            }
            fem_mesh->positions[picked_particle] = picked_position();
            fem_mesh->velocity[picked_particle].clear();

            break;
        case 1: // left click
        case 2: // right click
        default:
            fem_mesh->externalForce.index = picked_particle;
            fem_mesh->externalForce.value = picked_position() - fem_mesh->positions[picked_particle];
            if (picking_mode == 1)
                fem_mesh->externalForce.value *= 1000;
            else
                fem_mesh->externalForce.value *= 10000;
            break;
        }
    }
    return picking_mode;
}

int stop_picking(int picking_mode, int x, int y, int bt, double obj_x, double obj_y, double obj_z)
{
    picked_origin = camera_position;
    picked_dir = TCoord(obj_x, obj_y, obj_z) - picked_origin;
    picked_dir.normalize();
    if (fem_mesh->externalForce.index == picked_particle)
        fem_mesh->externalForce.index = -1;
    picked_particle = -1;
    if (fixed_particle != -1)
    {
        fem_mesh->removeFixedParticle(fixed_particle);
        fixed_particle = -1;
    }
    return -1;
}

//// MAIN METHOD ////

void render_gui()
{
	glDisable(GL_DEPTH_TEST);

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	gluOrtho2D(0.5f, window_width+0.5f, 0.5f, window_height+0.5f);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
    glEnable(GL_BLEND);
    float y0 = 20;
    float y1 = y0;
    if (background_color[0] > 0.5f)
        glColor4f(0.0f,0.0f,0.0f,1.0f);
    else
        glColor4f(1.0f,1.0f,1.0f,1.0f);
    if (render_flag[RENDER_FLAG_HELP] || render_flag[RENDER_FLAG_STAT])
    {
        drawTextFont(15);
        drawTextAlign(1,0);
        drawTextPos(window_width-20,y1-1);
        drawText("GPU Computing Gems - interactive FEM simulation using CUDA\n");
        drawTextPos(window_width-20,y1);
        y1 += drawText("GPU Computing Gems - interactive FEM simulation using CUDA\n");
    }
    if (render_flag[RENDER_FLAG_HELP])
    {
        drawTextAlign(1,0);
        drawTextPos(window_width-20,y1);
        glColor4f(0.2f,0.2f,0.2f,0.66f);
        drawTextFont(11);
        y1 += drawText(
        "To move the camera or apply force to a particle, clic and drag with the MOUSE\n"
        "To start/stop the simulation, press SPACE\n"
        "To change rendering mode, press F3\n"
        "To show/hide particles, press F4\n"
        "To show/hide this text, press F1\n"
        "See readme.txt for more instructions\n"
        );
    }
    if (render_flag[RENDER_FLAG_STAT])
    {
        if (background_color[0] > 0.5f)
            glColor4f(0.0f,0.0f,0.0f,1.0f);
        else
            glColor4f(1.0f,1.0f,1.0f,1.0f);
        drawTextFont(15);
        drawTextAlign(0,0);
        if (fps > 0)
        {
            drawTextPos(10,y0-1);
            drawTextF("FPS: %.1f\n",fps);
            drawTextPos(10,y0);
            y0 += drawTextF("FPS: %.1f\n",fps);
        }
        else
        {
            drawTextPos(10,y0);
            y0 += drawText("\n");
        }
        drawTextFont(13);
        y0 += drawTextF("%s\n",device_name.c_str());

        drawTextFont(11);
        if (fem_mesh)
        {
            y0 += drawTextF("# particles: %d\n",fem_mesh->positions.size());
            y0 += drawTextF("# elements: %d\n",fem_mesh->tetrahedra.size());
        }
        if (simulation_cg_iter)
        {
            y0 += drawTextF("CG iterations: %d\n",simulation_cg_iter);
        }
    }

    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
}
