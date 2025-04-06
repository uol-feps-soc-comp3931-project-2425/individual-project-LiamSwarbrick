#version 460 core

in vec3 color;

out vec4 frag_color;

#define INV_GAMMA (1.0 / 2.2)

void
main()
{
    // frag_color = vec4(color, 1.0);
    frag_color = vec4(pow(color, vec3(INV_GAMMA)), 1.0);
}
