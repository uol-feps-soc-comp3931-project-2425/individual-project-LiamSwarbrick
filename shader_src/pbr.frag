#version 460 core

layout(early_fragment_tests) in;

in vec3 frag_position_viewspace;
in vec2 texcoord_0;
in mat3 tbn_matrix;

layout (location = 0) out vec4 frag_color;

// Set once per frame
layout (location = 3) uniform vec3 sun_direction_viewspace;
layout (location = 4) uniform float sun_intensity;
layout (location = 5) uniform vec3 sun_color;

layout (location = 6) uniform float constant_attenuation;
layout (location = 7) uniform float linear_attenuation;
layout (location = 8) uniform float quadratic_attenuation;

struct PointLight
{
    vec4 position_xyz_range_w;
    vec4 color_rgb_intensity_a;
};

layout (std430, binding = 0) restrict buffer point_light_ssbo
{
    PointLight point_lights[];
};

// TODO: Remove MAX_NGON since I'm not clipping in the old way anymore...
//       Leaving note here for now so I can maybe mention this in writeup
#define MAX_NGON 15  // NOTE: This is the max ngon size after clipping (which can introduce more vertices)
                     // A size of 15 can handle a worst case clipping of a 10-gon - derivation in my writeup.
                     // A decagon can produce a star shaped area light similar to the figure in the Heitz paper.
// const int MAX_UNCLIPPED_NGON = (3 * MAX_NGON) / 2;
#define MAX_UNCLIPPED_NGON 10
struct AreaLight
{
    vec4 color_rgb_intensity_a;
    int n;
    int is_double_sided;
    float _packing0, _packing1;
    vec4 aabb_min;
    vec4 aabb_max;
    vec4 sphere_of_influence_center_xyz_radius_w;
    vec4 points_viewspace[MAX_UNCLIPPED_NGON];  // 4th component unused, vec3[] would be packed the same way but vec3 is implemented wrong on some drivers
};

layout (std430, binding = 2) restrict buffer area_light_ssbo
{
    AreaLight area_lights[];
};

layout (binding = 0) uniform sampler2D base_color_linear_space;
layout (binding = 1) uniform sampler2D metallic_roughness_texture;
layout (binding = 2) uniform sampler2D emissive_texture;
layout (binding = 3) uniform sampler2D occlusion_texture;
layout (binding = 4) uniform sampler2D normal_texture;

// LTC textures
layout (binding = 5) uniform sampler2D LTC1_texture;
layout (binding = 6) uniform sampler2D LTC2_texture;
const float LUT_SIZE = 64.0;  // dimensions of the LTC textures
const float LUT_SCALE = (LUT_SIZE - 1.0) / LUT_SIZE;
const float LUT_BIAS = 0.5 / LUT_SIZE;

// PBR material parameters
layout (location = 10) uniform vec4 base_color_factor;
layout (location = 11) uniform float metallic_factor;
layout (location = 12) uniform float roughness_factor;
layout (location = 13) uniform vec3 emissive_factor;
layout (location = 14) uniform float alpha_mask_cutoff;
layout (location = 15) uniform int is_normal_mapping_enabled;
layout (location = 16) uniform int is_alpha_blending_enabled;

#ifdef ENABLE_CLUSTERED_SHADING
    #ifndef CLUSTER_MAX_LIGHTS
        #define CLUSTER_MAX_LIGHTS 100
    #endif
    struct Cluster
    {
        vec4 min_point;
        vec4 max_point;
        uint point_count;
        uint area_count;
        uint point_indices[CLUSTER_MAX_LIGHTS/2];
        uint area_indices[CLUSTER_MAX_LIGHTS/2];
    };

    layout (std430, binding = 1) restrict buffer cluster_ssbo
    {
        Cluster clusters[];
    };

    // Clustered shading parameters
    layout (location = 17) uniform float near;
    layout (location = 18) uniform float far;
    layout (location = 19) uniform uvec3 grid_size;
    layout (location = 20) uniform uvec2 screen_dimensions;
#else
    layout (location = 9) uniform uint num_point_lights;
    layout (location = 21) uniform uint num_area_lights;
#endif  // ENABLE_CLUSTERED_SHADING


#define M_PI 3.1415926535897932384626433832795
#define INV_GAMMA (1.0 / 2.2)

vec3
pbr_metallic_roughness_brdf(vec4 base_color, float metallic, float roughness, vec3 V, vec3 L, vec3 N, vec3 H)
{
    // Implementation of lighting model from glTF BRDF appendix (https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#complete-model)
    // - V is the normalized vector from the shading location to the eye
    // - L is the normalized vector from the shading location to the light
    // - N is the surface normal in the same space as the above values (view space)
    // - H is the half vector, where H = normalize(L + V)

    // float HdotL = max(0., dot(H, L));
    // float HdotV = max(0., dot(H, V));
    // float NdotH = max(0., dot(N, H));
    // float NdotL = max(0., dot(N, L));
    // float NdotV = max(0., dot(N, V));

    float HdotL = dot(H, L);
    float HdotV = dot(H, V);
    float NdotH = dot(N, H);
    float NdotL = dot(N, L);
    float NdotV = dot(N, V);

    vec3 black = vec3(0.);
    vec3 c_diff = mix(base_color.rgb, black, metallic);
    vec3 f0 = mix(vec3(0.04), base_color.rgb, metallic);
    float alpha = roughness * roughness;
    float alpha_squared = alpha*alpha;

    vec3 F = f0 + (vec3(1.) - f0) * pow(1. - abs(HdotV), 5.);

    float Viz_numerator_0 = step(0., HdotL);
    float Viz_numerator_1 = step(0., HdotV);
    float Viz_denominator_0 = abs(NdotL) + sqrt(alpha_squared + (1. - alpha_squared) * NdotL * NdotL);
    float Viz_denominator_1 = abs(NdotV) + sqrt(alpha_squared + (1. - alpha_squared) * NdotV * NdotV);
    float Viz = (Viz_numerator_0 * Viz_numerator_1) / (Viz_denominator_0 * Viz_denominator_1);

    float D_numerator = alpha_squared * step(0., NdotH);
    float D_denominator = NdotH * NdotH * (alpha_squared - 1.) + 1.;
    float D = D_numerator / (M_PI * D_denominator * D_denominator);

    vec3 f_diffuse = (1. - F) * (1. / M_PI) * c_diff;
    vec3 f_specular = F * Viz * D;

    vec3 material = f_diffuse + f_specular;
    return material;
}


// The available area lights implementation only uses quadrilaterals and manually checks every edge case
// Hence a general algorithm was needed to perform clipping for any n-gon => Sutherland-Hodgman algorithm
/*
Why is Sutherland-Hodgman algorithm acceptable?
- It works with clipping non-convex polygons
- We only want one polygon returned, overlapping edges is okay for rendering (not for shadows as per the wiki) and we won't input crazy polygons anyway.
- Our 'clipping polygon' is simply the z>=0 half space, which means our inner loop has only one edge check. Simple and efficient.
*/
// void
// clip_polygon_to_hemisphere(vec4 points[MAX_UNCLIPPED_NGON], int in_n, out vec3 outpoints[MAX_NGON], out int out_n)
// {
//     // Run through the Sutherland–Hodgman algorithm to clip to the half-space z >= 0.0
//     out_n = 0;

//     for (int i = 0; i < in_n; ++i)
//     {
//         // int j = (i + 1) % in_n;
//         int j = i + 1;
//         if (j == in_n)
//         {
//             j = 0;
//         }
//         vec3 current = points[i].xyz;
//         vec3 next = points[j].xyz;

//         bool current_inside = current.z >= 0.0;
//         bool next_inside = next.z >= 0.0;

//         // 4 cases (including nothing)
//         if (current_inside && next_inside)
//         {
//             outpoints[out_n++] = next;
//         }
//         else if (current_inside != next_inside)
//         {
//             // intersecting_point = p1 + t*(p2-p1) = (x,y,0)^T implies t = z1 / (z1 - z2))
//             float t = current.z / (current.z - next.z);
//             vec3 intersect = mix(current, next, t);
//             outpoints[out_n++] = intersect;

//             if (next_inside)
//             {
//                 outpoints[out_n++] = next;
//             }
//         }
//     }
// }

vec3
integrate_edge_sector_vec(vec3 point_i, vec3 point_j)
{
    // Original code:
    // float x = acos(dot(point_i, point_j));
    // float cross_z = normalize(cross(point_i, point_j)).z;
    // return x * cross_z;

    float x = dot(point_i, point_j);
    float y = abs(x);
    float a = 0.8543985 + (0.4965155 + 0.0145206 * y) * y;
    float b = 3.4175940 + (4.1616724 + y) * y;
    float v = a / b;
    float theta_sintheta;
    if (x > 0.0)
    {
        theta_sintheta = v;
    }
    else
    {
        theta_sintheta = 0.5 * inversesqrt(max(1.0 - x*x, 1e-7)) - v;
    }
    
    return cross(point_i, point_j) * theta_sintheta;
}

vec3
integrate_lambertian_hemisphere(vec3 points[MAX_NGON], int n)
{
    // TODO: Consider unrolling this loop for quad, and clipped quad=pentagon
    vec3 vsum = integrate_edge_sector_vec(points[n-1], points[0]);  // Start with the wrap around pair (n-1, 0)
    for (int i = 0; i < n-1; ++i)
    {
        vsum += integrate_edge_sector_vec(points[i], points[i+1]);
    }

    return vsum;
}

// float
// integrate_edge_disk_sector_projection(vec3 point_i, vec3 point_j)
// {
//     // TODO: I recall somewhere that they may have used a cubic alternative to acos for better results
//     float x = acos(dot(point_i, point_j));
//     float cross_z = normalize(cross(point_i, point_j)).z;
//     return x * cross_z;
// }

// float
// integrate_lambertian_hemisphere(vec3 points[MAX_NGON], int n)
// {
//     // TODO: Consider unrolling this loop for quad, and clipped quad=pentagon
//     float sum = integrate_edge_disk_sector_projection(points[n-1], points[0]);  // Start with the wrap around pair (n-1, 0)
//     for (int i = 0; i < n-1; ++i)
//     {
//         sum += integrate_edge_disk_sector_projection(points[i], points[i+1]);
//     }

//     return sum;
// }

/*
N := fragment normal vector
V := fragment to camera vector
P := fragment viewspace position
Minv := The transformation from a clamped cosine to the linearly transformed cosine
*/
vec3
LTC_evaluate(vec3 N, vec3 V, vec3 P, mat3 Minv, vec4 viewspace_points[MAX_UNCLIPPED_NGON], int viewspace_points_n, bool double_sided)
{
    // Construct tangent space of orthonormal basis vectors x,y,z=T1,T2,N
    vec3 T1 = normalize(V - N * dot(V, N));
    vec3 T2 = cross(N, T1);

    // Move LTC domain change matrix into the tangent space.
    Minv = Minv * transpose(mat3(T1, T2, N));
    
    // Is fragment behind polygon
    vec3 dir = viewspace_points[0].xyz - P;
    vec3 light_normal = cross(viewspace_points[1].xyz - viewspace_points[0].xyz, viewspace_points[2].xyz - viewspace_points[0].xyz);
    bool behind = dot(dir, light_normal) < 0.;

    // Move polygon into tangent space
    for (int i = 0; i < viewspace_points_n; ++i)  // If i only used quad area lights I could unroll this but I'm not
    {
        viewspace_points[i].xyz = Minv * (viewspace_points[i].xyz - P);
    }

    vec3 points_o[MAX_NGON];  // P_o in the paper
    int n;
    // clip_polygon_to_hemisphere(viewspace_points, viewspace_points_n, points_o, n);
    
    // // Degenerate case: Very important as it stops away faces from having very negative values in double sided
    // if (n < 3)
    //     return vec3(0.0);
    
    // for (int i = 0; i < n; ++i)
    //     points_o[i] = normalize(points_o[i]);
    

    // Uncomment this when commenting clipping code
    for (int i = 0; i < viewspace_points_n; ++i)
        points_o[i] = normalize(viewspace_points[i].xyz);
    n = viewspace_points_n;

    vec3 vsum = integrate_lambertian_hemisphere(points_o, n);
    float len = length(vsum);
    float z = vsum.z / len;
    if (behind)
    {
        z = -z;
    }
    
    vec2 uv = vec2(z * 0.5 + 0.5, len);  // Move from range [-1,1] to [0, 1]
    uv = uv * LUT_SCALE + LUT_BIAS;

    // Fetch horizon clipping form factor
    float scale = texture(LTC2_texture, uv).w;

    float sum = len * scale;
    if (!behind && !double_sided)
    {
        sum = 0.0;
    }

    // Outgoing radiance from fragment from the polygon
    vec3 Lo_i = vec3(sum);
    return Lo_i;
}

void
main()
{
    vec4 base_color = texture(base_color_linear_space, texcoord_0) * base_color_factor;
    float alpha = base_color.a;
    if (alpha < alpha_mask_cutoff)
    {
        discard;
    }
    else if (is_alpha_blending_enabled == 0)
    {
        // Support opaque alpha mode in glTF
        alpha = 1.0;
    }

    vec4 metallic_roughness = texture(metallic_roughness_texture, texcoord_0) * vec4(0., roughness_factor, metallic_factor, 0.);
    float metallic = metallic_roughness.b;
    float roughness = metallic_roughness.g;

    vec3 emissive = texture(emissive_texture, texcoord_0).rgb * emissive_factor.rgb;
    float occlusion = texture(occlusion_texture, texcoord_0).r;

    vec3 N;
    if (is_normal_mapping_enabled == 1)
    {
        vec3 normal_sample = texture(normal_texture, texcoord_0).rgb * 2.0 - vec3(1.0);
        N = normalize(tbn_matrix * normal_sample);
    }
    else
    {
        N = normalize(tbn_matrix[2]);  // Set N to view_normal
    }

    vec3 V = normalize(-frag_position_viewspace);  // Since in viewspace camera position is at the origin (cam-fragpos)

    // Directional light (The Sun)
    vec3 L_sun = normalize(sun_direction_viewspace);
    vec3 H_sun = normalize(L_sun + V);
    float NdotL_sun = max(dot(N, L_sun), 0.);
    vec3 sun_radiance = pbr_metallic_roughness_brdf(base_color, metallic, roughness, V, L_sun, N, H_sun)
        * NdotL_sun * sun_color * sun_intensity;
    
/* Adding shit to make figure START:*/
    // sun_radiance = simple_brdf()
    // vec3 col = base_color.xyz;

    // frag_color = vec4(pow(col, vec3(INV_GAMMA)), alpha);
    // return;
/* END */
    
    // Lights (TODO: Put area lights in here as well)
    vec3 sum_pl_radiance = vec3(0.);
    vec3 sum_arealight_radiance = vec3(0.0);  // TODO

#ifdef ENABLE_CLUSTERED_SHADING
    // Find fragment's cluster by solving cluster equation for index
    /*
    For reference we want ID.z from:
    float cluster_near_plane = near * pow(clipping_ratio, float(ID.z) / grid_depth);
    float cluster_far_plane = near * pow(clipping_ratio, float(ID.z + 1) / grid_depth);
    */
    uint tile_z = uint((log(abs(frag_position_viewspace.z) / near) * grid_size.z) / log(far / near));
    // vec2 tile_size = screen_dimensions / grid_size.xy;
    vec2 tile_size = ceil(screen_dimensions / vec2(grid_size.xy));
    uvec3 tile = uvec3(gl_FragCoord.xy / tile_size, tile_z);
    uint tile_index = tile.x + (tile.y * grid_size.x) + (tile.z * grid_size.x * grid_size.y);
// TODO: if tile_z is far cluster then fade out
    uint num_point_lights = clusters[tile_index].point_count;
    uint num_area_lights = clusters[tile_index].area_count;
    
    // if (tile_z == grid_size.z - 1)
    // {
    //     // Special far cluster lighting system?
    // }
    // else

    // Point lights
    for (int i = 0; i < num_point_lights; ++i)
    {
        uint light_index = clusters[tile_index].point_indices[i];
#else
    for (int light_index = 0; light_index < num_point_lights; ++light_index)
    {
#endif  // ENABLE_CLUSTERED_SHADING

        PointLight pl = point_lights[light_index];

        // Unpack point_lights[i]
        vec3  pl_viewpos   = pl.position_xyz_range_w.xyz;
        float pl_range     = pl.position_xyz_range_w.w;
        vec3  pl_color     = pl.color_rgb_intensity_a.rgb;
        float pl_intensity = pl.color_rgb_intensity_a.a;
        
        // Calculate directions for BRDF
        vec3 L_point = normalize(pl_viewpos - frag_position_viewspace);
        vec3 H_point = normalize(L_point + V);
        float NdotL_point = max(dot(N, L_point), 0.);
        
        // Attenuation with inverse square law
        float dist = distance(pl_viewpos, frag_position_viewspace);
        float attenuation = 1.0 / (constant_attenuation + linear_attenuation * dist + quadratic_attenuation * dist * dist);
        
        // Inverse square law with a hard cutoff after range
        attenuation = max(0.0, 1.0 - (dist / pl_range));
        attenuation *= attenuation;

        vec3 pl_radiance = pbr_metallic_roughness_brdf(base_color, metallic, roughness, V, L_point, N, H_point)
            * NdotL_point * pl_color * pl_intensity * attenuation;

        sum_pl_radiance += pl_radiance;
    }

    // Area Lights
#ifdef ENABLE_CLUSTERED_SHADING
    #pragma unroll(CLUSTER_MAX_LIGHTS/2)
    for (int i = 0; i < num_area_lights; ++i)
    {
        uint light_index = clusters[tile_index].area_indices[i];
#else
    for (int light_index = 0; light_index < num_area_lights; ++light_index)
    {
#endif  // ENABLE_CLUSTERED_SHADING
        AreaLight al = area_lights[light_index];

        float dot_NV = clamp(dot(N, V), 0.0, 1.0);
        vec2 ltc_uv = vec2(roughness, sqrt(1.0 - dot_NV));
        ltc_uv = ltc_uv * LUT_SCALE + LUT_BIAS;

        vec4 t1 = texture(LTC1_texture, ltc_uv);
        vec4 t2 = texture(LTC2_texture, ltc_uv);

        mat3 Minv = mat3(
            vec3(t1.x,  0., t1.y),
            vec3(  1.,  1.,   0.),
            vec3(t1.z, 0.,  t1.w)
        );

        // NOTE: al.viewspace_points is a vec4 array but can pass to a vec3 array due to having the same padding.
        // if (al.n == 3) sum_arealight_radiance += vec3(10.0);
        vec3 diffuse = LTC_evaluate(N, V, frag_position_viewspace, mat3(1), al.points_viewspace, al.n, al.is_double_sided == 1);
        vec3 specular = LTC_evaluate(N, V, frag_position_viewspace, Minv, al.points_viewspace, al.n, al.is_double_sided == 1);

        // GGX BRDF shadowing and Fresnel
        // t2.x: shadowedF90 (F90 normally should be 1.0)
        // t2.y: Smith function for Geometric Attenuation Term, it is dot(V or L, H).
        vec3 F0 = mix(vec3(0.04), base_color.rgb, metallic);
        specular *= F0 * t2.x + (1.0 - F0) * t2.y;
        // specular = vec3(0.0);
        // diffuse = vec3(0.0);
        sum_arealight_radiance += al.color_rgb_intensity_a.a * al.color_rgb_intensity_a.rgb * (specular + base_color.rgb * diffuse);
    }

    // Add ambient light
    vec3 ambient_color = vec3(0.00);
    vec3 ambient = occlusion * ambient_color * base_color.rgb;

    // Get final color in linear space
    vec3 final_linear_color =
        sun_radiance +
        sum_pl_radiance +
        sum_arealight_radiance +
        emissive + ambient;
    // vec3 final_linear_color = specular*vec3(0.0, 1.0, 0.0) + diffuse*vec3(0.0, 0.0, 1.0);

// NOTE: I still have the define TRANSPARENT_PASS for when I need to add OIT

#ifdef SHOW_NORMALS
    // frag_color = vec4(metallic_roughness.rgb, alpha);
    // frag_color = mix(vec4(N, alpha), vec4(metallic_roughness.rgb, alpha), 0.5);

    float amount_red = float(num_point_lights/10.0);
    float amount_blue = float(num_area_lights/5.0);
    float amount_green = metallic_roughness.g * 0.3;
    // float amount_red = float(num_point_lights/CLUSTER_MAX_LIGHTS);
    
    frag_color = vec4(amount_red, amount_green, amount_blue, alpha);
#else
    // Gamma correction: Radiance is linear space, we convert it to sRGB for the display.
    frag_color = vec4(pow(final_linear_color, vec3(INV_GAMMA)), alpha);
#endif


    // final_linear_color = emissive.rgb;
    // frag_color = vec4(light_radiance, alpha);
}
