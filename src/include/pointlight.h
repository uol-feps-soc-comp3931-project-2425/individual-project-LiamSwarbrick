#ifndef POINTLIGHT_H
#define POINTLIGHT_H

#include <cglm/cglm.h>

typedef struct PointLight
{
    // Don't worry the glsl std430 version of PointLight doesn't use vec3s
    vec3 position;
    float _padding0;  // w component here is used for range but only range is calculated when uploading to shader

    vec3 color;
    float intensity;
}
PointLight;

float calculate_point_light_range(
    float point_light_intensity,
    float min_perceivable_intensity,
    float q, float l, float c);

#endif  // POINTLIGHT_H
