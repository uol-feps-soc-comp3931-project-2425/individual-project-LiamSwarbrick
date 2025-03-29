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
make_area_light(vec3 position, vec3 normal_vector, int is_double_sided, int n, float hue, float intensity, float width, float height)
{
    // When hue is negative we pick a random color
    // When width is negative we choose random dimensions

    AreaLight al;

    vec3 scale_vector = { width, height, 1.0f };
    if (hue < 0.0f)
    {
        hue = rng_rangef(0.0f, 1.0f);
        intensity = rng_rangef(3.0, 25.0f);
    }

    if (width < 0.0f)
    {
        scale_vector[0] = rng_rangef(0.3f, 3.0f);
        if (n != 6)
        {
            scale_vector[1] = rng_rangef(0.3f, 3.0f);
        }
        else
        {
            // Scale uniformly for star shape.
            scale_vector[1] = scale_vector[0];
        }
    }
    
    vec3 rgb; hsv_to_rgb(hue, 1.0f, 1.0f, rgb);
    al.color_rgb_intensity_a[0] = rgb[0];
    al.color_rgb_intensity_a[1] = rgb[1];
    al.color_rgb_intensity_a[2] = rgb[2];
    al.color_rgb_intensity_a[3] = intensity;// rng_rangef(3.0f, 10.0f);

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
    glm_scale(scale, scale_vector);

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
polygon_area(AreaLight* al)
{
    // TODO: Maybe instead of using 3D AABB surface area as upper bound for polygon area,
    // find the polygons plane with the first 3 points (since n >= 3) then project onto 2D
    // and use the 2D AABB (could be a tighter upper bound but more costly cpu side which might be bad)

    if (al->n < 3) return 0.0f;

    // Find normal from first 3 points
    vec3 u, v, normal;
    glm_vec3_sub(al->points_worldspace[1], al->points_worldspace[0], u);
    glm_vec3_sub(al->points_worldspace[2], al->points_worldspace[0], v);
    glm_vec3_cross(u, v, normal);
    glm_vec3_normalize(normal);

    // Construct an orthonormal basis
    vec3 tangent, bitangent;
    if (fabsf(normal[0]) > fabsf(normal[1])) {
        glm_vec3_cross((vec3){0,1,0}, normal, tangent);  // Try y-up first
    } else {
        glm_vec3_cross((vec3){1,0,0}, normal, tangent);  // Otherwise, x-right
    }
    glm_vec3_normalize(tangent);
    glm_vec3_cross(normal, tangent, bitangent);

    // Project the polygon onto the new 2D basis
    vec2 projected[MAX_UNCLIPPED_NGON];
    for (int i = 0; i < al->n; ++i) {
        vec3 p;
        glm_vec3_sub(al->points_worldspace[i], al->points_worldspace[0], p);
        projected[i][0] = glm_vec3_dot(p, tangent);
        projected[i][1] = glm_vec3_dot(p, bitangent);
    }

    // Compute the area using the Shoelace theorem
    float area = 0.0f;
    for (int i = 0; i < al->n; ++i) {
        int j = (i + 1) % al->n;
        area += projected[i][0] * projected[j][1] - projected[j][0] * projected[i][1];
    }

    return fabsf(area) * 0.5f;

    // // Compute AABB of polygon
    // vec3 min; glm_vec3_copy(al->points_worldspace[0], min);
    // vec3 max; glm_vec3_copy(al->points_worldspace[0], max);
    // for (int i = 1; i < al->n; ++i)
    // {
    //     glm_vec3_minv(min, al->points_worldspace[i], min);
    //     glm_vec3_maxv(max, al->points_worldspace[i], max);
    // }

    // // Compute surface area of AABB which is an upper bound for the polygons area
    // vec3 size;
    // glm_vec3_sub(max, min, size);

    // return 2.0f * (size[0] * size[1] + size[1] * size[2] + size[2] * size[0]);
}

float
calculate_area_light_influence_radius(AreaLight* al, float area, float min_perceivable)
{
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
    float r = al->color_rgb_intensity_a[0];
    float b = al->color_rgb_intensity_a[1];
    float g = al->color_rgb_intensity_a[2];
    float intensity = al->color_rgb_intensity_a[3];
    float luminance = 0.2126f * r + 0.7152f * g + 0.0722f * b;
    float flux = luminance * intensity * area;
    // float flux = ((r + b + g) / 3.0f) * intensity * area;

    if (al->is_double_sided) flux *= 2.0f;

    // Inverse square law with hemispherical falloff adjustment
    return 2.0f* sqrtf(flux / (2.0f * M_PI * min_perceivable));
}
