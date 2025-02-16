#version 460 core

layout (location = 0) in vec3 v_position;
layout (location = 1) in vec3 v_normal;
layout (location = 2) in vec2 v_texcoord_0;
layout (location = 3) in vec4 v_tangent;

// Set once per draw call
layout (location = 0) uniform mat4 mvp;
layout (location = 1) uniform mat4 model_view;
layout (location = 2) uniform mat4 normal_matrix;

out vec3 frag_position_viewspace;
out vec2 texcoord_0;
out mat3 tbn_matrix;
out vec3 sun_direction_viewspace;

void
main()
{
    // Calculate TBN matrix for normal mapping
    vec3 view_normal = normalize(mat3(normal_matrix) * v_normal);
    vec3 view_tangent = normalize(mat3(normal_matrix) * vec3(v_tangent));
    vec3 view_bitangent = normalize(cross(view_normal, view_tangent) * v_tangent.w);
    tbn_matrix = mat3(view_tangent, view_bitangent, view_normal);

    frag_position_viewspace = (model_view * vec4(v_position, 1.0)).xyz;  // Outgoing light direction from fragment view space to camera
    texcoord_0 = v_texcoord_0;
    gl_Position = mvp * vec4(v_position, 1.0);
}
