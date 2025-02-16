#version 460 core

layout (location = 0) in vec3 v_position;
layout (location = 1) in vec3 v_color;

// Set once per draw call
layout (location = 0) uniform mat4 mvp;

out vec3 color;

void
main()
{
    gl_Position = mvp * vec4(v_position, 1.0);
    color = v_color;
}
