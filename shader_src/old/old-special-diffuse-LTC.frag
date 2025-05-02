
#if 0  // Couldn't get diffuse approximation to work, it wouldn't be much faster anyway...
vec3
integrate_edge_sector_vec_diffuse(vec3 point_i, vec3 point_j)
{
    // Cheaper quadratic good enough for diffuse
    float x = dot(point_i, point_j);
    float y = abs(x);
    float theta_sintheta = 1.5708 + (-0.879406 + 0.308609 * y) * y;
    if (x < 0.0)
    {
        theta_sintheta = M_PI * inversesqrt(1.0 - x*x) - theta_sintheta;
    }
    return cross(point_i, point_j) * theta_sintheta;
}

vec3
integrate_lambertian_hemisphere_diffuse(vec3 points[MAX_UNCLIPPED_NGON], int n)
{
    vec3 vsum = integrate_edge_sector_vec_diffuse(points[n-1], points[0]);  // Start with the wrap around pair (n-1, 0)
    for (int i = 0; i < n-1; ++i)
    {
        vsum += integrate_edge_sector_vec_diffuse(points[i], points[i+1]);
    }
    return vsum;
}

vec3
LTC_evaluate_diffuse(vec3 N, vec3 V, vec3 P, vec4 viewspace_points[MAX_UNCLIPPED_NGON], int viewspace_points_n, bool double_sided)
{
    #ifdef COUNT_LIGHT_OPS
    atomicCounterIncrement(light_ops_atomic_counter_buffer);
    #endif  // COUNT_LIGHT_OPS

    // Early exit if fragment is behind polygon
    vec3 dir = viewspace_points[0].xyz - P;
    vec3 light_normal = cross(viewspace_points[1].xyz - viewspace_points[0].xyz, viewspace_points[2].xyz - viewspace_points[0].xyz);
    bool behind = dot(dir, light_normal) < 0.;
    if (!behind && !double_sided)
    {
        return vec3(0.0);
    }

    // Minv for this one is just identity in local coordinate system
    vec3 T1 = normalize(V - N * dot(V, N));
    vec3 T2 = cross(N, T1);
    mat3 Minv = transpose(mat3(T1, T2, N));  // Move LTC domain change matrix into the tangent space.

    // transformation, and using slightly faster quadratic approximation for arccos
    vec3 points_o[MAX_UNCLIPPED_NGON];
    for (int i = 0; i < viewspace_points_n; ++i)
    {
        points_o[i] = normalize(Minv * (viewspace_points[i].xyz - P));
    }
    vec3 vsum = integrate_lambertian_hemisphere_diffuse(points_o, viewspace_points_n);
    float len = length(vsum);
    float z = vsum.z / len;
    if (behind)
    {
        z = -z;
    }
    
    vec2 uv = vec2(z * 0.5 + 0.5, len);  // Move from range [-1,1] to [0, 1]
    uv = uv * LUT_SCALE + LUT_BIAS;

    // Fetch horizon clipped norm of the transformed BRDF me thinks
    float horizon_clipped_scale = texture(LTC2_texture, uv).w;
    float sum = len * horizon_clipped_scale;

    // Outgoing radiance from fragment from the polygon
    vec3 Lo_i = vec3(sum);
    return Lo_i;
}
#endif
