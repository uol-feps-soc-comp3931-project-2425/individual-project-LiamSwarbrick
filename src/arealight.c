#include "arealight.h"

#include "basic_types.h"
#include <stdlib.h>
#include <string.h>

void
transform_area_light(AreaLight* al, mat4 transform)
{
    for (int i = 0; i < al->n; ++i)
    {
        glm_mat4_mulv(transform, al->points_worldspace[i], al->points_worldspace[i]);
    }
}

AreaLight
make_area_light(vec3 position, vec3 normal_vector, int is_double_sided, int n)
{
    AreaLight al;
    vec3 rgb; hsv_to_rgb(rng_rangef(0.0f, 1.0f), 1.0f, 1.0f, rgb);
    al.color_rgb_intensity_a[0] = rgb[0];
    al.color_rgb_intensity_a[1] = rgb[1];
    al.color_rgb_intensity_a[2] = rgb[2];
    al.color_rgb_intensity_a[3] = rng_rangef(3.0f, 10.0f);

    al.n = n;
    al.is_double_sided = is_double_sided;

    if (n == 3)
    {
        memcpy(al.points_worldspace, triangle_points, sizeof(triangle_points));
    }
    else if (n == 4)
    {
        memcpy(al.points_worldspace, quad_points, sizeof(quad_points));
    }
    else if (n == 5)
    {
        memcpy(al.points_worldspace, pentagon_points, sizeof(pentagon_points));
    }
    else if (n == 10)
    {
        memcpy(al.points_worldspace, star_points, sizeof(star_points));
    }
    else
    {
        assert(0 && "Invalid n for area lights.");
    }
    
    mat4 scale = GLM_MAT4_IDENTITY_INIT;
    if (n != 6)
        glm_scale(scale, (vec3){ rng_rangef(0.3f, 2.0f), rng_rangef(0.3f, 2.0f), rng_rangef(0.3f, 2.0f) });

    mat4 rot = GLM_MAT4_IDENTITY_INIT;
    {
        // Rotating the plane (with normal (0,0,-1) to plane with normal (normal_vector))
        vec3 old_normal = { 0.0f, 0.0f, -1.0f };
        vec3 rot_axis;
        glm_cross(old_normal, normal_vector, rot_axis);
        float rot_angle = acosf(glm_dot(old_normal, normal_vector));
        
        glm_rotate_atm(rot, (vec3){ 0.0f, 0.0f, 0.0f }, rot_angle, rot_axis);
    }

    mat4 move;
    glm_translate_make(move, position);

    mat4 transform = GLM_MAT4_IDENTITY_INIT;
    glm_mul(move, rot, transform);
    glm_mul(transform, scale, transform);

    transform_area_light(&al, transform);

    // The transform uses homogeneous coordinates but I simply rely on x,y,z in the shader instead of w
    // so in order for the rendered light sources to match the location of the light reflections we must make w=1...
    for (int i = 0; i < n; ++i)
    {
        float w = al.points_worldspace[i][3];
        al.points_worldspace[i][0] /= w;
        al.points_worldspace[i][1] /= w;
        al.points_worldspace[i][2] /= w;
        al.points_worldspace[i][3] = 1.0f;
    }

    return al;
}


float
polygon_area_aabb_upper_bound(AreaLight* al)
{
    /*
    An upper bound area approximation is need to predict the radius of the
    sphere of influence approximation for the area light.
    Upper bound is necessary because we don't want to under-assign to clusters
    during light culling since that would cause visual artefacts.
    */

    // TODO: Maybe instead of using 3D AABB surface area as upper bound for polygon area,
    // find the polygons plane with the first 3 points (since n >= 3) then project onto 2D
    // and use the 2D AABB (could be a tighter upper bound but more costly cpu side which might be bad)

    if (al->n < 3) return 0.0f;

    // Compute AABB of polygon
    vec3 min; glm_vec3_copy(al->points_worldspace[0], min);
    vec3 max; glm_vec3_copy(al->points_worldspace[0], max);
    for (int i = 1; i < al->n; ++i)
    {
        glm_vec3_minv(min, al->points_worldspace[i], min);
        glm_vec3_maxv(max, al->points_worldspace[i], max);
    }

    // Compute surface area of AABB which is an upper bound for the polygons area
    vec3 size;
    glm_vec3_sub(max, min, size);

    return 2.0f * (size[0] * size[1] + size[1] * size[2] + size[2] * size[0]);
}

float
calculate_area_light_influence_radius(AreaLight* al, float area, float min_perceivable)
{
    float r = al->color_rgb_intensity_a[0];
    float b = al->color_rgb_intensity_a[1];
    float g = al->color_rgb_intensity_a[2];
    float intensity = al->color_rgb_intensity_a[3];
    float flux = ((r + b + g) / 3.0f) * intensity * area;

    // Inverse square law with hemispherical falloff adjustment
    return sqrtf(flux / (2.0f * M_PI * min_perceivable)) * (al->is_double_sided ? 1.0f : 2.0f);

    /* Math notes:
    For sphere E = flux/(4*pi*r^2) for distance r from a point light
    For an area light the emission is hemispherical: E= flux / (2*pi*r^2)
    NOTE: The is_double_sided property is simply used to emit the same light in both directions, we don't
        normalize our radius according to it.

    Solve for r:
    min_perceivable_intensity = flux / (2 * pi * r^2)
    r^2 = flux/(2 * pi * min_perceivable)
    r = sqrt(...)
    */
}
