#include "pointlight.h"

float
calculate_point_light_range(float point_light_intensity, float min_perceivable_intensity,
    float q, float l, float c)
{
    /* Attenuation Formula given constants and range r:
    *  min_intensity = source_intensity / (c + l*r + q*r*r)
    *  
    *  Rearrange for r.
    *  source_intensity/min_intensity = c + l*r + q*r*r
    *  qr^2 + lr + (c-source_intensity/min_intensity) = 0
    *
    *  Quadratic formula to get range that minimum intensity is hit (positive solution)
    *  r = (-l + sqrt(l^2 - 4*q*(c-source_intensity/min_intensity)) )/(2q)
    */
    float discriminant = l*l - 4.0f * q * (c - point_light_intensity / min_perceivable_intensity);
    
    float point_light_range;
    if (discriminant >= 0.0f)
    {
        point_light_range = (-l + sqrtf(discriminant)) / (2.0f * q);
    }
    else
    {
        point_light_range = 0.0f;
    }

    return point_light_range;
}
