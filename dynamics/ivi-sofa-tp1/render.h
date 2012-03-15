#ifndef RENDER_H
#define RENDER_H

// methods from render.cpp

void initgl();
void render();
void shaders_reload();

// methods from gui.cpp

void render_gui();
void reset_camera();
void move_camera(int dx, int dy, int mode);
int start_picking(int picking_mode, int x, int y, int bt, double obj_x, double obj_y, double obj_z);
int move_picking(int picking_mode, int x, int y, int dx, int dy, double obj_x, double obj_y, double obj_z);
int stop_picking(int picking_mode, int x, int y, int bt, double obj_x, double obj_y, double obj_z);
int update_picking(int picking_mode);

enum {
    RENDER_FLAG_HELP = 0,
    RENDER_FLAG_STAT,
    RENDER_FLAG_MESH,
    RENDER_FLAG_POINT,
    RENDER_FLAG_DEBUG,
    RENDER_FLAG_SHADERS,
    RENDER_NFLAGS
};

struct RenderFlag
{
    int value;
    int nvalues;
    const char * name;
    RenderFlag(int val, int nval, const char* name)
    : value(val), nvalues(nval), name(name)
    {}
    RenderFlag(bool val, const char* name)
    : value(val), nvalues(2), name(name)
    {}
    operator int() const { return value; }
    bool operator==(int i) const { return value == i; }
    bool operator!=(int i) const { return value != i; }
    void operator=(int i) { value = i % nvalues; }
    void operator=(bool b) { value = b ? 1 : 0; }
    void operator++() { value = (value + 1) % nvalues; }
    void operator++(int) { value = (value + 1) % nvalues; }    
};
extern RenderFlag render_flag[RENDER_NFLAGS]; // rendering flags as controlled by F1...F5;

extern TCoord camera_lookat;
extern double camera_distance;
extern TCoord camera_direction;
extern TCoord camera_position;
extern TColor background_color;


#endif
