#ifndef AREALIGHT_H
#define AREALIGHT_H

#include <cglm/cglm.h>

#define MAX_UNCLIPPED_NGON 10

typedef struct AreaLight
{
    vec4 color_rgb_intensity_a;
    int n;
    int is_double_sided;
    float _packing0, _packing1;

    // For clustered shading
    vec4 min_point;
    vec4 max_point;
    vec4 sphere_of_influence_center_xyz_radius_w;
    
    vec4 points_worldspace[MAX_UNCLIPPED_NGON];
}
AreaLight;

// defined in arealight_shape_data.c
extern const float triangle_points[12];
extern const float quad_points[16];
extern const float pentagon_points[20];
extern const float star_points[40];
extern const unsigned int star_indices[24];

void transform_area_light(AreaLight* al, mat4 transform);
AreaLight make_area_light(vec3 position, vec3 normal_vector, int is_double_sided, int n, float hue);
float polygon_area_aabb_upper_bound(AreaLight* al);
float calculate_area_light_influence_radius(AreaLight* al, float area, float min_perceivable);

#endif  // AREALIGHT_H
