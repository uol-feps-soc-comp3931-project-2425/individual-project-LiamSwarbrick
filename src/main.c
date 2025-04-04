// Dependencies:
// glfw    - Windowing and OpenGL context
// glad    - OpenGL 4.6 core function loader
// cglm    - Inline linear algebra
// cgltf   - Parses GLTF file format into C structs
// stb     - Image file loading
// nuklear - Immediate mode GUI

#include <glad/glad.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#define CGLTF_IMPLEMENTATION
#include <cgltf.h>
#include <cglm/cglm.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

// Include the ANSI C IM GUI library "nuklear" with GLFW-OpenGL4 backend
#define NK_INCLUDE_FIXED_TYPES
#define NK_INCLUDE_STANDARD_IO
#define NK_INCLUDE_STANDARD_VARARGS
#define NK_INCLUDE_DEFAULT_ALLOCATOR
#define NK_INCLUDE_VERTEX_BUFFER_OUTPUT
#define NK_INCLUDE_FONT_BAKING
#define NK_INCLUDE_DEFAULT_FONT
#define NK_IMPLEMENTATION
#define NK_GLFW_GL3_IMPLEMENTATION
#include "Nuklear/nuklear.h"
#include "Nuklear/demo/common/style.c"
#include "Nuklear/demo/glfw_opengl3/nuklear_glfw_gl3.h"
#define NUKLEAR_MAX_VERTEX_BUFFER 512 * 1024
#define NUKLEAR_MAX_ELEMENT_BUFFER 128 * 1024


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "basic_types.h"
#include "pointlight.h"
#include "arealight.h"
#include "ltc_matrix.h"

#include "point_light_data.h"

typedef struct FreeCamera
{
    vec3 pos;
    f32 pitch;
    f32 yaw;

    f32 fov_y;
    f32 near_plane;
    f32 far_plane;
    u32 width;
    u32 height;
    mat4 camera_matrix;  // = projection * view
    mat4 view_matrix;
    mat4 projection_matrix;
}
FreeCamera;
// Collision detection can always be added with a colliding function move(camera, scene, move_vector)

enum GlobalBindingPoints
{
    GLOBAL_SSBO_INDEX_POINTLIGHTS = 0,
    GLOBAL_SSBO_INDEX_CLUSTERGRID = 1,
    GLOBAL_SSBO_INDEX_AREALIGHTS  = 2,
};

enum PBRShaderLocations
{
    PBR_LOC_mvp                        =0,
    PBR_LOC_model_view                 =1,
    PBR_LOC_normal_matrix              =2,
    PBR_LOC_sun_direction_viewspace    =3,
    PBR_LOC_sun_intensity              =4,
    PBR_LOC_sun_color                  =5,
    PBR_LOC_constant_attenuation       =6,
    PBR_LOC_linear_attenuation         =7,
    PBR_LOC_quadratic_attenuation      =8,

    PBR_LOC_num_point_lights           =9,

    PBR_LOC_base_color_factor          =10,
    PBR_LOC_metallic_factor            =11,
    PBR_LOC_roughness_factor           =12,
    PBR_LOC_emissive_factor            =13,
    PBR_LOC_alpha_mask_cutoff          =14,
    PBR_LOC_is_normal_mapping_enabled  =15,
    PBR_LOC_is_alpha_blending_enabled  =16,

    // Clustered shading params
    PBR_LOC_near              =17,
    PBR_LOC_far               =18,
    PBR_LOC_grid_size         =19,
    PBR_LOC_screen_dimensions =20,

    PBR_LOC_num_area_lights =21,
};

enum PBR_Shader_Texture_Units
{
    PBR_TEXUNIT_base_color_linear_space     =0,
    PBR_TEXUNIT_metallic_roughness_texture  =1,
    PBR_TEXUNIT_emissive_texture            =2,
    PBR_TEXUNIT_occlusion_texture           =3,
    PBR_TEXUNIT_normal_texture              =4,

    PBR_NUM_USED_TEXTURE_UNITS
};

#define TEXUNIT_LTC1_texture 5
#define TEXUNIT_LTC2_texture 6
#define TEXUNIT_cluster_normals_cubemap 7
#define TEXUNIT_representative_normals_texture 8

u32
gl_component_type_from_cgltf(cgltf_component_type component_type)
{
    assert(component_type != cgltf_component_type_invalid && "glTF file contained invalid component type");
    switch (component_type)
    {
        case cgltf_component_type_r_8: return GL_BYTE; break;
        case cgltf_component_type_r_8u: return GL_UNSIGNED_BYTE; break;
        case cgltf_component_type_r_16: return GL_SHORT; break;
        case cgltf_component_type_r_16u: return GL_UNSIGNED_SHORT; break;
        case cgltf_component_type_r_32u: return GL_UNSIGNED_INT; break;
        case cgltf_component_type_r_32f: return GL_FLOAT; break;
        default:
            assert(0 && "cgltf parsing bug occurred, invalid component type.");
            return 0;
    }
}

u32
gl_primitive_mode_from_cgltf(cgltf_primitive_type primitive_type)
{
    switch (primitive_type)
    {
        case cgltf_primitive_type_points:         return GL_POINTS;
        case cgltf_primitive_type_lines:          return GL_LINES;
        case cgltf_primitive_type_line_loop:      return GL_LINE_LOOP;
        case cgltf_primitive_type_line_strip:     return GL_LINE_STRIP;
        case cgltf_primitive_type_triangles:      return GL_TRIANGLES;
        case cgltf_primitive_type_triangle_strip: return GL_TRIANGLE_STRIP;
        case cgltf_primitive_type_triangle_fan:   return GL_TRIANGLE_FAN;
        default:
            assert(0 && "cgltf parsing bug occurred, invalid primitive type.");
            return 0;
    }
}

typedef struct VAO_Attributes { b8 has_position, has_texcoord_0, has_normal, has_tangent; } VAO_Attributes;
typedef struct VAO_Range { u32 begin; u32 count; } VAO_Range;

#define INTEGRATED_GPU
#ifdef INTEGRATED_GPU
    #define CLUSTER_GRID_SIZE_X 32//32//16 
    #define CLUSTER_GRID_SIZE_Y 32//32//9
    #define CLUSTER_GRID_SIZE_Z 16//32
#else
    #define CLUSTER_GRID_SIZE_X 16//24
#define CLUSTER_GRID_SIZE_Y 9//16
#define CLUSTER_GRID_SIZE_Z 12//16
#endif  // INTEGRATED_GPU
#define CLUSTER_NORMALS_COUNT 1//1//24//54//6   // of the form 6*n*n, e.g. 6, 24, 54  // 1 disables normal clustering
#define NUM_CLUSTERS (CLUSTER_GRID_SIZE_X * CLUSTER_GRID_SIZE_Y * CLUSTER_GRID_SIZE_Z * CLUSTER_NORMALS_COUNT)
#define CLUSTER_DEFAULT_MAX_LIGHTS 200

typedef struct  ClusterMetaData
{  // Manually padded so size is same as the std430 glsl struct Cluster
    vec4 min_point;
    vec4 max_point;
    u32 point_count;
    u32 area_count;
    f32 _padding[2];
    
    // The Cluster data on the GPU also stores
    // - u32 light_indices[CLUSTER_MAX_LIGHTS/2]
    // - u32 area_indices[CLUSTER_MAX_LIGHTS/2]
    // - u32 area_light_flags[CLUSTER_MAX_LIGHTS/2]
}
ClusterMetaData;

typedef struct Scene
{
    // NOTE:
    // - white_texture used for materials with no texture
    // - vao_range usage: first primitive for meshes[mesh_index] at vao_range[mesh_index].begin

    cgltf_data* data;
    u32* buffer_objects;
    u32* texture_objects;
    u32 white_texture;
    u32 flat_normal_texture;
    u32* vaos;
    u32 vaos_count;
    VAO_Attributes* vaos_attributes;  // Disable normal mapping when a vao has no tangents
    VAO_Range* vao_ranges;
    u32 total_opaque_primitives;
    u32 total_transparent_primitives;

    // Single Directional Light:
    vec3 sun_direction;
    f32 sun_intensity;
    vec3 sun_color;

    // Point Light attenuation parameters
    float attenuation_constant;
    float attenuation_linear;
    float attenuation_quadratic;
    float minimum_perceivable_intensity;  // Threshold for light range calculations
#define ATTENUATION_CONSTANT_DEFAULT 1.0f
#define ATTENUATION_LINEAR_DEFAULT    2.5f//0.9f
#define ATTENUATION_QUADRATIC_DEFAULT 5.0f//4.0f
#define MINIMUM_PERCEIVABLE_INTENSITY_DEFAULT 0.015f//0.05

    // Area light clustered shading parameters
    float param_roughness;
    float param_min_intensity;
    float param_intensity_saturation;
}
Scene;

typedef struct PBRMaterialUniforms
{
    vec4 base_color_factor;
    f32 metallic_factor;
    f32 roughness_factor;
    vec3 emissive_factor;
    f32 alpha_mask_cutoff;
    s32 is_normal_mapping_enabled;
    s32 is_alpha_blending_enabled;
}
PBRMaterialUniforms;

typedef struct Loaded_Image
{
    u8* pixels;
    int width;
    int height;
}
Loaded_Image;

typedef struct PBRDrawCall
{
    cgltf_primitive* prim;
    u32 vao;
    b32 double_sided;
    u32 primitive_mode;  // e.g. GL_TRIANGLES

    // Uniform data
    mat4 mvp;
    mat4 model_view;
    mat4 normal_matrix;
    PBRMaterialUniforms uniforms;

    // Textures
    u32 texture_ids[PBR_NUM_USED_TEXTURE_UNITS];
}
PBRDrawCall;

Loaded_Image
load_image(const char* filename, cgltf_image* image)
{
    Loaded_Image loaded = { 0 };

    // We don't need the image->mime_type field since stb_image.h automatically determines the file type
    if (image->buffer_view)  // Image file stored in glTF buffer
    {
        printf("Problem with %s, Currently not supporting image buffer views in glTF files soz.\n", filename);
        exit(1);
    }
    else if (image->uri)
    {
        // Not supporting embedded images in glTF files
        if (strncmp(image->uri, "data:", 5) == 0)
        {
            // In future could add something similar to https://github.com/google/filament/blob/main/libs/gltfio/src/ResourceLoader.cpp#L133
            printf("Problem with %s, Currently not supporting embedded images in glTF files soz.\n", filename);
            exit(1);
        }

        // The uri is relative to the glTF file so we must adjust it to get the actual path
        // Get directory path to where image uri starts
        char* dir_path;
        const char* last_slash = strrchr(filename, '/');
        if (last_slash == NULL)
        {
            dir_path = strdup("./");
        }
        else
        {
            // Get length of directory including last '/'
            size_t dir_len = last_slash - filename + 1;
            dir_path = calloc(1, dir_len + 1);
            strncpy(dir_path, filename, dir_len);  // <- strncpy isn't null terminated but calloc means it ends in 0 automatically
        }

        char* image_path;
        image_path = calloc(1, strlen(dir_path) + strlen(image->uri) + 1);
        strcpy(image_path, dir_path);
        strcat(image_path, image->uri);
        free(dir_path);

        // printf("Loading image buffer from file: %s\n", image_path);
        int comp;
        loaded.pixels = stbi_load(image_path, &loaded.width, &loaded.height, &comp, 4);
        if (loaded.pixels == NULL)
        {
            printf("Error loading image %s, exiting\n", image_path);
            exit(1);
        }

        free(image_path);
    }
    else
    {
        assert(0 && "Invalid glTF format encountered when loading image");  // Invalid glTF format.
    }

    return loaded;
}

void
free_loaded_image(Loaded_Image* image)
{
    if (image->pixels)
    {
        free(image->pixels);
        image->pixels = NULL;
    }
}

Scene
load_gltf_scene(const char* filename)
{
    // Load GLTF file with cgltf
    cgltf_data* data = NULL;
    {
        cgltf_options options = { 0 };
        cgltf_result result = cgltf_parse_file(&options, filename, &data);

        if (result == cgltf_result_success)
        {
            result = cgltf_load_buffers(&options, data, filename);
        }

        if (result != cgltf_result_success)
        {
            printf("Failed to load gltf file: %s\n", filename);
            exit(1);
        }
    }
    
    // Create buffer objects
    u32* buffers = calloc(data->buffers_count, sizeof(u32));
    {
        glCreateBuffers(data->buffers_count, buffers);

        // Upload data to each created buffer
        for (u32 i = 0; i < data->buffers_count; ++i)
        {
            cgltf_buffer* buffer = &data->buffers[i];

            glNamedBufferStorage(
                buffers[i],
                buffer->size,
                buffer->data,
                GL_MAP_WRITE_BIT
            );
        }
    }
    
    // Create textures and load images from file
    u32* textures = calloc(data->textures_count, sizeof(u32));
    {
        u32* image_textures = calloc(data->images_count, sizeof(u32));
        for (u32 img_i = 0; img_i < data->images_count; ++img_i)
        {
            cgltf_image* image = &data->images[img_i];

            // Load images used for base colors or emissives as sRGB instead of linear space
            int is_srgb = 0;
            for (u32 mat_i = 0; mat_i < data->materials_count; ++mat_i)
            {
                cgltf_material* material = &data->materials[mat_i];
                if (material->pbr_metallic_roughness.base_color_texture.texture &&
                    material->pbr_metallic_roughness.base_color_texture.texture->image == image)
                {
                    is_srgb = 1;
                    break;
                }

                if (material->emissive_texture.texture &&
                    material->emissive_texture.texture->image == image)
                {
                    is_srgb = 1;
                    break;
                }
            }

            Loaded_Image loaded_img = load_image(filename, image);
            u32 tex;
            u32 internal_format = is_srgb ? GL_SRGB8_ALPHA8 : GL_RGBA8;
            
            glCreateTextures(GL_TEXTURE_2D, 1, &tex);
            glTextureStorage2D(tex, 1, internal_format, loaded_img.width, loaded_img.height);
            glTextureSubImage2D(tex, 0, 0, 0, loaded_img.width, loaded_img.height, GL_RGBA, GL_UNSIGNED_BYTE, loaded_img.pixels);

            // Default texture filtering
            glTextureParameteri(tex, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTextureParameteri(tex, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTextureParameteri(tex, GL_TEXTURE_WRAP_S, GL_REPEAT);
            glTextureParameteri(tex, GL_TEXTURE_WRAP_T, GL_REPEAT);

            glGenerateTextureMipmap(tex);

            image_textures[img_i] = tex;

            free_loaded_image(&loaded_img);
        }

        // Map textures to images
        for (u32 tex_i = 0; tex_i < data->textures_count; ++tex_i)
        {
            cgltf_texture* texture = &data->textures[tex_i];
            textures[tex_i] = image_textures[texture->image - data->images];

            cgltf_sampler* sampler = texture->sampler;
            if (sampler)
            {
                u32 tex = textures[tex_i];
                glTextureParameteri(tex, GL_TEXTURE_MIN_FILTER, sampler->min_filter ? sampler->min_filter : GL_LINEAR);
                glTextureParameteri(tex, GL_TEXTURE_MAG_FILTER, sampler->mag_filter ? sampler->mag_filter : GL_LINEAR);
                glTextureParameteri(tex, GL_TEXTURE_WRAP_S, sampler->wrap_s ? sampler->wrap_s : GL_REPEAT);
                glTextureParameteri(tex, GL_TEXTURE_WRAP_T, sampler->wrap_t ? sampler->wrap_t : GL_REPEAT);
            }
        }

        free(image_textures);
    }
    
    // Create white texture for non-textured materials
    u32 white_texture;
    {
        u8 single_white_pixel_data[4] = { 255, 255, 255, 255 };
        glCreateTextures(GL_TEXTURE_2D, 1, &white_texture);
        glTextureParameteri(white_texture, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTextureParameteri(white_texture, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTextureParameteri(white_texture, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTextureParameteri(white_texture, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTextureStorage2D(white_texture, 1, GL_RGBA8, 1, 1);
        glTextureSubImage2D(white_texture, 0, 0, 0, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, single_white_pixel_data);
    }

    // Create flat normal map for materials with normal maps
    u32 flat_normal_texture;
    {
        u8 single_flat_normal_pixel_data[3] = { 128, 128, 255 };
        glCreateTextures(GL_TEXTURE_2D, 1, &flat_normal_texture);
        glTextureParameteri(flat_normal_texture, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTextureParameteri(flat_normal_texture, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTextureParameteri(flat_normal_texture, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTextureParameteri(flat_normal_texture, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTextureStorage2D(flat_normal_texture, 1, GL_RGB8, 1, 1);
        glTextureSubImage2D(flat_normal_texture, 0, 0, 0, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, single_flat_normal_pixel_data);
    }
    
    // Create vertex arrays
    DynamicArray vaos = create_array(1 * sizeof(u32));
    DynamicArray vaos_attributes = create_array(1 * sizeof(VAO_Attributes));
    VAO_Range* vao_ranges = calloc(data->meshes_count, sizeof(VAO_Range));
    u32 total_opaque_primitives = 0;
    u32 total_transparent_primitives = 0;

    for (u32 mesh_i = 0; mesh_i < data->meshes_count; ++mesh_i)
    {
        cgltf_mesh* mesh = &data->meshes[mesh_i];

        // Keep track of the range of elements in vaos that this mesh uses
        vao_ranges[mesh_i].begin = array_length(&vaos, sizeof(u32));  // Used for rendering
        vao_ranges[mesh_i].count = mesh->primitives_count;                  // ..

        // Create VAO for each primitive in this mesh:
        u32* mesh_vaos = push_size(&vaos, sizeof(u32), mesh->primitives_count);
        glCreateVertexArrays(mesh->primitives_count, mesh_vaos);

        VAO_Attributes* mesh_vaos_attributes = push_size(&vaos_attributes, sizeof(VAO_Attributes), mesh->primitives_count);
        memset(mesh_vaos_attributes, 0, sizeof(VAO_Attributes) * mesh->primitives_count);
        
        // Now loop over each primitive's vao and set attributes
        for (u32 prim_i = 0; prim_i < mesh->primitives_count; ++prim_i)
        {
            cgltf_primitive* prim = &mesh->primitives[prim_i];
            u32 vao = mesh_vaos[prim_i];

            // Keep track of transparent and opaque primitive counts for fast alpha blend sorting during rendering
            if (prim->material)
            {
                if (prim->material->alpha_mode == cgltf_alpha_mode_blend)
                {
                    ++total_transparent_primitives;
                }
                else
                {
                    ++total_opaque_primitives;
                }
            }
            else
            {
                ++total_opaque_primitives;
            }

            // Find the attributes POSITION, NORMAL, and TEXCOORD_0
            cgltf_attribute* POSITION = NULL;
            cgltf_attribute* NORMAL = NULL;
            cgltf_attribute* TEXCOORD_0 = NULL;
            cgltf_attribute* TANGENT = NULL;
            for (u32 attrib_i = 0; attrib_i < prim->attributes_count; ++attrib_i)
            {
                cgltf_attribute* attrib = &prim->attributes[attrib_i];
                switch (attrib->type)
                {
                    case cgltf_attribute_type_position:
                        POSITION = attrib;
                        break;

                    case cgltf_attribute_type_normal:
                        NORMAL = attrib;
                        break;
                    
                    case cgltf_attribute_type_texcoord:
                        if (attrib->index == 0)  // GLTF can have multiple texcoords
                        {
                            TEXCOORD_0 = attrib;
                        }
                        break;
                    
                    case cgltf_attribute_type_tangent:
                        TANGENT = attrib;
                        break;
                    
                    default:
                        continue;
                }
            }
            
            if (POSITION == NULL)
            {
                printf("Error: POSITION attribute required but missing in %s\n", filename);
                exit(1);
            }

            // Find the index of each attribute buffer in data->buffers
            // This index also corresponds to the vbo in the u32* buffers array.
            int POSITION_vbo_index = -1;
            int NORMAL_vbo_index = -1;
            int TEXCOORD_0_vbo_index = -1;
            int TANGENT_vbo_index = -1;

            for (u32 buffer_i = 0; buffer_i < data->buffers_count; ++buffer_i)
            {
                // Remember multiple attributes can have the same buffer
                cgltf_buffer* buffer = &data->buffers[buffer_i];

                if (POSITION && buffer == POSITION->data->buffer_view->buffer)
                {
                    POSITION_vbo_index = buffer_i;
                    mesh_vaos_attributes[prim_i].has_position = 1;
                }

                if (NORMAL && buffer == NORMAL->data->buffer_view->buffer)
                {
                    NORMAL_vbo_index = buffer_i;
                    mesh_vaos_attributes[prim_i].has_normal = 1;
                }

                if (TEXCOORD_0 && buffer == TEXCOORD_0->data->buffer_view->buffer)
                {
                    TEXCOORD_0_vbo_index = buffer_i;
                    mesh_vaos_attributes[prim_i].has_texcoord_0 = 1;
                }

                if (TANGENT && buffer == TANGENT->data->buffer_view->buffer)
                {
                    TANGENT_vbo_index = buffer_i;
                    mesh_vaos_attributes[prim_i].has_tangent = 1;
                }
            }
            
            // Make sure if the attribute exists we found the buffer for it
            assert((POSITION   && POSITION_vbo_index   != -1) || !POSITION);
            assert((NORMAL     && NORMAL_vbo_index     != -1) || !NORMAL);
            assert((TEXCOORD_0 && TEXCOORD_0_vbo_index != -1) || !TEXCOORD_0);
            assert((TANGENT    && TANGENT_vbo_index    != -1) || !TANGENT);

            // Assign seperate binding points for each attribute.
            // Some glTF files use the same buffer for multiple attributes, but it's still fine in OpenGL to bind the same buffer to multiple binding points.
            const u32 POSITION_binding_point = 0;
            const u32 NORMAL_binding_point = 1;
            const u32 TEXCOORD_0_binding_point = 2;
            const u32 TANGENT_binding_point = 3;

            // Enable OpenGL vertex attributes and set attribute binding points and formats
            // Then Assign the VBO of the attribute to the to the VAO
            //  NOTE: Vertex format issues helped a little with this post: https://community.khronos.org/t/clarity-of-accessor-offset-and-glvertexarrayattribformat/106240

            // Tangents stored as vec4s, the w component should be the bitangent sign: https://blender.stackexchange.com/questions/220756/why-does-blender-output-vec4-tangents-for-gltf

            const u32 POSITION_attrib_index   = 0;  // layout (location = 0) in vec3 v_position;
            const u32 NORMAL_attrib_index     = 1;  // layout (location = 1) in vec3 v_normal;
            const u32 TEXCOORD_0_attrib_index = 2;  // layout (location = 2) in vec2 v_texcoord_0;
            const u32 TANGENT_attrib_index    = 3;  // layout (location = 3) in vec4 v_tangent;

            if (POSITION && POSITION->data->type == cgltf_type_vec3)// && POSITION->data->component_type == cgltf_component_type_r_32f)
            {
                u32 vbo_POSITION = buffers[POSITION_vbo_index];
                u32 POSITION_buffer_offset = POSITION->data->offset + POSITION->data->buffer_view->offset;
                u32 POSITION_component_type = gl_component_type_from_cgltf(POSITION->data->component_type);

                glEnableVertexArrayAttrib(vao, POSITION_attrib_index);
                glVertexArrayAttribBinding(vao, POSITION_attrib_index, POSITION_binding_point);
                glVertexArrayAttribFormat(vao, POSITION_attrib_index, 3, POSITION_component_type, GL_FALSE, 0);
                glVertexArrayVertexBuffer(vao, POSITION_binding_point, vbo_POSITION, POSITION_buffer_offset, POSITION->data->stride);
                // NOTE: cgltf's accessor stride isn't just copied from buffer view stride,
                // if buffer_view stride is zero (meaning tightly packed), accessor stride is calculated by cgltf based on types instead.
                // This is useful because glVertexArrayVertexBuffer doesn't support stride of 0 to mean tight packing.

                // TODO: Also works, maybe a better idea?
                // glVertexArrayAttribFormat(
                //     vao,
                //     POSITION_attrib_index,
                //     3,
                //     POSITION_component_type,
                //     GL_FALSE,
                //     POSITION->data->offset
                // );
                // glVertexArrayVertexBuffer(
                //     vao,
                //     POSITION_binding_point,
                //     vbo_POSITION,
                //     POSITION->data->buffer_view->offset,
                //     POSITION->data->stride
                // );
            }
            else if (POSITION)
            {
                printf("Error loading %s\nPOSITION attribute: Unsupported attribute format", filename);
                exit(1);
            }

            if (NORMAL && NORMAL->data->type == cgltf_type_vec3)// && NORMAL->data->component_type == cgltf_component_type_r_32f)
            {
                u32 vbo_NORMAL = buffers[NORMAL_vbo_index];
                u32 NORMAL_buffer_offset = NORMAL->data->offset + NORMAL->data->buffer_view->offset;
                u32 NORMAL_component_type = gl_component_type_from_cgltf(NORMAL->data->component_type);

                glEnableVertexArrayAttrib(vao, NORMAL_attrib_index);
                glVertexArrayAttribBinding(vao, NORMAL_attrib_index, NORMAL_binding_point);
                glVertexArrayAttribFormat(vao, NORMAL_attrib_index, 3, NORMAL_component_type, GL_FALSE, 0);
                glVertexArrayVertexBuffer(vao, NORMAL_binding_point, vbo_NORMAL, NORMAL_buffer_offset, NORMAL->data->stride);
            }
            else if (NORMAL)
            {
                printf("Error loading %s\nNORMAL attribute: Unsupported attribute format", filename);
                exit(1);
            }

            if (TEXCOORD_0 && TEXCOORD_0->data->type == cgltf_type_vec2)// && TEXCOORD_0->data->component_type == cgltf_component_type_r_32f)
            {
                u32 vbo_TEXCOORD_0 = buffers[TEXCOORD_0_vbo_index];
                u32 TEXCOORD_0_buffer_offset = TEXCOORD_0->data->offset + TEXCOORD_0->data->buffer_view->offset;
                u32 TEXCOORD_0_component_type = gl_component_type_from_cgltf(TEXCOORD_0->data->component_type);

                glEnableVertexArrayAttrib(vao, TEXCOORD_0_attrib_index);
                glVertexArrayAttribBinding(vao, TEXCOORD_0_attrib_index, TEXCOORD_0_binding_point);
                glVertexArrayAttribFormat(vao, TEXCOORD_0_attrib_index, 2, TEXCOORD_0_component_type, GL_FALSE, 0);
                glVertexArrayVertexBuffer(vao, TEXCOORD_0_binding_point, vbo_TEXCOORD_0, TEXCOORD_0_buffer_offset, TEXCOORD_0->data->stride);
            }
            else if (TEXCOORD_0)
            {
                printf("Error loading %s\nTEXCOORD_0 attribute: Unsupported attribute format", filename);
                exit(1);
            }

            if (TANGENT && TANGENT->data->type == cgltf_type_vec4)// && TANGENT->data->component_type == cgltf_component_type_r_32f)
            {
                u32 vbo_TANGENT = buffers[TANGENT_vbo_index];
                u32 TANGENT_buffer_offset = TANGENT->data->offset + TANGENT->data->buffer_view->offset;
                u32 TANGENT_component_type = gl_component_type_from_cgltf(TANGENT->data->component_type);

                glEnableVertexArrayAttrib(vao, TANGENT_attrib_index);
                glVertexArrayAttribBinding(vao, TANGENT_attrib_index, TANGENT_binding_point);
                glVertexArrayAttribFormat(vao, TANGENT_attrib_index, 4, TANGENT_component_type, GL_FALSE, 0);
                glVertexArrayVertexBuffer(vao, TANGENT_binding_point, vbo_TANGENT, TANGENT_buffer_offset, TANGENT->data->stride);
            }
            else if (TANGENT)
            {
                printf("Error loading %s\nTANGENT attribute: Unsupported attribute format: %d, %d\n", filename, TANGENT->data->type, TANGENT->data->component_type);
                exit(1);
            }

            // Get the primitive's indices if it has them
            if (prim->indices != NULL)
            {
                int ebo_index = -1;
                for (u32 buffer_i = 0; buffer_i < data->buffers_count; ++buffer_i)
                {
                    if (&data->buffers[buffer_i] == prim->indices->buffer_view->buffer)
                    {
                        ebo_index = buffer_i;
                        break;
                    }
                }

                assert(ebo_index >= 0);  // Make sure we found the buffer
                u32 ebo = buffers[ebo_index];
                glVertexArrayElementBuffer(vao, ebo);
            }

            // printf("Loaded the following attributes: ");
            // if (POSITION)
            //     printf("positions");
            // if (TEXCOORD_0)
            //     printf(", texcoord_0s");
            // if (NORMAL)
            //     printf(", normals");
            // if (TANGENT)
            //     printf(", tangents");
            // printf("\n");
        }
    }

    Scene scene;
    scene.data = data;
    scene.buffer_objects = buffers;
    scene.texture_objects = textures;
    scene.white_texture = white_texture;
    scene.flat_normal_texture = flat_normal_texture;
    scene.vaos = vaos.data_buffer;
    scene.vaos_count = vaos.used_size / sizeof(u32);
    scene.vaos_attributes = vaos_attributes.data_buffer;
    scene.vao_ranges = vao_ranges;
    scene.total_opaque_primitives = total_opaque_primitives;
    scene.total_transparent_primitives = total_transparent_primitives;

    scene.attenuation_constant   = ATTENUATION_CONSTANT_DEFAULT;
    scene.attenuation_linear     = ATTENUATION_LINEAR_DEFAULT;
    scene.attenuation_quadratic  = ATTENUATION_QUADRATIC_DEFAULT;
    scene.minimum_perceivable_intensity = MINIMUM_PERCEIVABLE_INTENSITY_DEFAULT;
    
    printf("Loaded glTF scene \"%s\"\n   - Number of VAOS: %d\n   - Number of textures: %d\n\n", filename, (int)array_length(&vaos, sizeof(u32)), (int)data->textures_count);

    return scene;
}


typedef struct Program
{
    GLFWwindow* window;
    u32 w, h;
    f32 aspect_ratio;
    b32 is_msaa_enabled;
    b32 is_minimized;
    f64 time;
    f32 dt;
    u64 frame_counter;
    const char* driver_name;

    struct nk_glfw gui_glfw;
    struct nk_context* gui_context;
    struct nk_font_atlas* gui_font_atlas;

    b32 render_as_wireframe;  // F1 to toggle
    b32 render_just_normals;  // F2 to toggle
    b32 is_clustered_shading_enabled;  // F3 to toggle
    u32 max_lights_per_cluster;

    b32 keydown_forward;
    b32 keydown_backward;
    b32 keydown_left;
    b32 keydown_right;
    b32 keydown_up;
    b32 keydown_down;
    b32 keydown_sprint;
    b32 keydown_zoom_in;
    b32 keytoggle_disable_gui;
    b32 mouse_capture_on;
    f64 mouse_relative_x;
    f64 mouse_relative_y;
    u32 last_number_key;

    u32 shader_area_light_polygons;
    u32 shader_pbr_opaque;
    // u32 shader_pbr_transparent;
    u32 shader_compute_clusters;
    u32 shader_light_assignment;

    // LTC1 and LTC2 contain matrices for transforming the clamped cosine distribution
    // to linearly transformed cosine distributions
    u32 LTC1_texture;  // Inverse M
    u32 LTC2_texture;  // (GGX norm, fresnel, 0(unused), sphere for horizon-clipping)

    FreeCamera cam;
    Scene scene;
    u32 default_shader;
    u32 current_shader;

    // Point light shader storage block (std430)
    u32 point_light_ssbo;
    u32 point_light_ssbo_max;
    #define SSBO_DEFAULT_MAX_POINT_LIGHTS 10000

    // Area light shader storage block (std430)
    u32 area_light_ssbo;
    u32 area_light_ssbo_max;
    #define SSBO_DEFAULT_MAX_AREA_LIGHTS 3000

    // Cluster grid SSBO
    u32 cluster_grid_ssbo;
    u32 cluster_normals_cubemap;  // get the quantized normal using a cubemap lookup.
    u32 representative_normals_1dtexure;  // the inverse of the cubemap (go from normal index to vector)

    // Atomic buffers
    b32 is_light_op_counting_enabled;
    u32 light_ops_atomic_counter_buffer;
    u32* light_ops_mapped_pointer;
    u32 last_light_ops_value;

    // Time query objects
    u32 compute_time_query;
    u64 compute_time_last_frame;

    // Dynamically add/change point lights in scene here
    DynamicArray point_lights;
    DynamicArray area_lights;
}
Program;

// For input use glfwGetKey

Program program = { 0 };

void
init_empty_cluster_grid()  // Abstraction to use when changing cluster settings at runtime
{
    // Init empty cluster grid of default size (any resizing happens in reload_shaders())
    if (program.cluster_grid_ssbo)
    {
        glDeleteBuffers(1, &program.cluster_grid_ssbo);
    }
    u32 cluster_size = sizeof(ClusterMetaData)  // header
        + sizeof(u32) * program.max_lights_per_cluster  // point lights & area light ids
        + sizeof(u32) * program.max_lights_per_cluster / 2;  // area light flags
    glCreateBuffers(1, &program.cluster_grid_ssbo);
    glNamedBufferData(program.cluster_grid_ssbo, cluster_size * NUM_CLUSTERS, NULL, GL_STATIC_COPY);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, GLOBAL_SSBO_INDEX_CLUSTERGRID, program.cluster_grid_ssbo);

    // TODO: Generate cube maps for cluster indexing based on quantized normals
#if CLUSTER_NORMALS_COUNT == 1
    printf("Cluster normals disabled: Generating dummy cubemap anyway.\n");
    int n = 1;  // This generates the 1x1x1 cube normal map, even though it isn't used
#else
    // CLUSTER_NORMALS_COUNT = 6*n*n
    assert(CLUSTER_NORMALS_COUNT % 6 == 0 && CLUSTER_NORMALS_COUNT > 0);
    int n = (int)sqrt(CLUSTER_NORMALS_COUNT / 6);  // e.g. n=3 gives a 3x3 cubemap which is 6 3x3 textures
#endif
    float n_f32 = (float)n;

    u32 cubemap_face_size = n*n * 3*sizeof(float);  // nxn texture of vec3s
    u32 index_cubemap_face_size = n*n * sizeof(u32);
    float* cubemap_texture_data = malloc(6 * cubemap_face_size);
        float* cubemap_posx = (float*)((u8*)cubemap_texture_data + 0*cubemap_face_size);
        float* cubemap_negx = (float*)((u8*)cubemap_texture_data + 1*cubemap_face_size);
        float* cubemap_posy = (float*)((u8*)cubemap_texture_data + 2*cubemap_face_size);
        float* cubemap_negy = (float*)((u8*)cubemap_texture_data + 3*cubemap_face_size); 
        float* cubemap_posz = (float*)((u8*)cubemap_texture_data + 4*cubemap_face_size);
        float* cubemap_negz = (float*)((u8*)cubemap_texture_data + 5*cubemap_face_size);
    u32* index_cubemap_data = malloc(6 * index_cubemap_face_size);
        u32* index_cubemap_posx = (u32*)((u8*)index_cubemap_data + 0*index_cubemap_face_size);
        u32* index_cubemap_negx = (u32*)((u8*)index_cubemap_data + 1*index_cubemap_face_size);
        u32* index_cubemap_posy = (u32*)((u8*)index_cubemap_data + 2*index_cubemap_face_size);
        u32* index_cubemap_negy = (u32*)((u8*)index_cubemap_data + 3*index_cubemap_face_size); 
        u32* index_cubemap_posz = (u32*)((u8*)index_cubemap_data + 4*index_cubemap_face_size);
        u32* index_cubemap_negz = (u32*)((u8*)index_cubemap_data + 5*index_cubemap_face_size);

    for (int v = 0; v < n; ++v)
    {
        for (int u = 0; u < n; ++u)
        {
            // Consider uv's spanning width/height of n units, (0,0) at bottom left,
            float u_val = (float)u + 0.5f;
            float v_val = (float)v + 0.5f;
            
            // and then move (0,0) to centre by adding (n/2, n/2)
            u_val -= 0.5f * n_f32;
            v_val -= 0.5f * n_f32;

            // and then divide to get vectors on the unit cube
            float inv_n = 1.0f / n_f32;
            u_val *= inv_n;
            v_val *= inv_n;
            
            // For u,v mappings see diagram here https://www.khronos.org/opengl/wiki/File:CubeMapAxes.png
            u32 vector_index = 3 * (n*v + u);
            f32 invsqrt = sqrtf(1.0f + u_val*u_val + v_val*v_val);  // Normalize to move from unit cube to unit sphere
            
            cubemap_posx[vector_index + 0] =  1.0f  * invsqrt;
            cubemap_posx[vector_index + 1] = -v_val * invsqrt;
            cubemap_posx[vector_index + 2] = -u_val * invsqrt;
            cubemap_negx[vector_index + 0] = -1.0f  * invsqrt;
            cubemap_negx[vector_index + 1] = -v_val * invsqrt;
            cubemap_negx[vector_index + 2] =  u_val * invsqrt;
            cubemap_posy[vector_index + 0] =  u_val * invsqrt;
            cubemap_posy[vector_index + 1] =  1.0f  * invsqrt;
            cubemap_posy[vector_index + 2] =  v_val * invsqrt;
            cubemap_negy[vector_index + 0] =  u_val * invsqrt;
            cubemap_negy[vector_index + 1] = -1.0f  * invsqrt;
            cubemap_negy[vector_index + 2] = -v_val * invsqrt;
            cubemap_posz[vector_index + 0] =  u_val * invsqrt;
            cubemap_posz[vector_index + 1] = -v_val * invsqrt;
            cubemap_posz[vector_index + 2] =  1.0f  * invsqrt;
            cubemap_negz[vector_index + 0] = -u_val * invsqrt;
            cubemap_negz[vector_index + 1] = -v_val * invsqrt;
            cubemap_negz[vector_index + 2] = -1.0f  * invsqrt;

            // Index = faceid*n*n + v*n + u
            index_cubemap_posx[n*v + u] = 0*n*n + v*n + u;
            index_cubemap_negx[n*v + u] = 1*n*n + v*n + u;
            index_cubemap_posy[n*v + u] = 2*n*n + v*n + u;
            index_cubemap_negy[n*v + u] = 3*n*n + v*n + u;
            index_cubemap_posz[n*v + u] = 4*n*n + v*n + u;
            index_cubemap_negz[n*v + u] = 5*n*n + v*n + u;


            // printf("%f %f %f,\n ", cubemap_posx[vector_index + 0], cubemap_posx[vector_index + 1], cubemap_posx[vector_index + 2]);
            // printf("%f %f %f,\n ", cubemap_negx[vector_index + 0], cubemap_negx[vector_index + 1], cubemap_negx[vector_index + 2]);
            // printf("%f %f %f,\n ", cubemap_posy[vector_index + 0], cubemap_posy[vector_index + 1], cubemap_posy[vector_index + 2]);
            // printf("%f %f %f,\n ", cubemap_negy[vector_index + 0], cubemap_negy[vector_index + 1], cubemap_negy[vector_index + 2]);
            // printf("%f %f %f,\n ", cubemap_posz[vector_index + 0], cubemap_posz[vector_index + 1], cubemap_posz[vector_index + 2]);
            // printf("%f %f %f,\n ", cubemap_negz[vector_index + 0], cubemap_negz[vector_index + 1], cubemap_negz[vector_index + 2]);
        }
    }

    printf("Cubemap Normal Data:\n");
    for (int i = 0; i < 6*n*n; ++i)
    {
        vec3 v; glm_vec3_copy(&cubemap_texture_data[3*i], v);
        printf("%d : %d : %f, %f, %f\n", i, index_cubemap_data[i], v[0], v[1], v[2]);
    }
    
    // Cubemap for quantizing normals
    glCreateTextures(GL_TEXTURE_CUBE_MAP, 1, &program.cluster_normals_cubemap);
    glTextureStorage2D(program.cluster_normals_cubemap, 1, GL_R32UI, n, n);
    for (u32 i = 0; i < 6; ++i)
    {
        glTextureSubImage3D(program.cluster_normals_cubemap, 0, 0, 0, i, n, n, 1, GL_RED_INTEGER, GL_UNSIGNED_INT,
            (u8*)index_cubemap_data + i*index_cubemap_face_size
        );
    }
    glTextureParameteri(program.cluster_normals_cubemap, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTextureParameteri(program.cluster_normals_cubemap, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTextureParameteri(program.cluster_normals_cubemap, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(program.cluster_normals_cubemap, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTextureParameteri(program.cluster_normals_cubemap, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

    // Normal index to quantized normal (1D texture equivalent of cubemap)
    glCreateTextures(GL_TEXTURE_1D, 1, &program.representative_normals_1dtexure);
    glTextureStorage1D(program.representative_normals_1dtexure, 1, GL_RGB32F, 6*n*n);
    glTextureSubImage1D(program.representative_normals_1dtexure, 0, 0, 6*n*n, GL_RGB, GL_FLOAT, cubemap_texture_data);

    glTextureParameteri(program.representative_normals_1dtexure, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTextureParameteri(program.representative_normals_1dtexure, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTextureParameteri(program.representative_normals_1dtexure, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);

    free(cubemap_texture_data);
    free(index_cubemap_data);
}

void
init_global_renderer_buffers()
{
    program.point_light_ssbo_max = SSBO_DEFAULT_MAX_POINT_LIGHTS;
    program.area_light_ssbo_max  = SSBO_DEFAULT_MAX_AREA_LIGHTS;

    //
    // We give each SSBO a different binding point so that we only have to bind them once on load
    //
    
    init_empty_cluster_grid();

    // Init empty point lights
    if (program.point_light_ssbo)
    {
        glDeleteBuffers(1, &program.point_light_ssbo);
    }
    glCreateBuffers(1, &program.point_light_ssbo);
    glNamedBufferData(program.point_light_ssbo, program.point_light_ssbo_max * sizeof(PointLight), NULL, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, GLOBAL_SSBO_INDEX_POINTLIGHTS, program.point_light_ssbo);

    // Init empty area lights
    if (program.area_light_ssbo)
    {
        glDeleteBuffers(1, &program.area_light_ssbo);
    }
    glCreateBuffers(1, &program.area_light_ssbo);
    glNamedBufferData(program.area_light_ssbo, program.area_light_ssbo_max * sizeof(AreaLight), NULL, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, GLOBAL_SSBO_INDEX_AREALIGHTS, program.area_light_ssbo);

    // Init atomic counter to profile number of light operations
    if (program.light_ops_atomic_counter_buffer)
    {
        if (program.light_ops_mapped_pointer)
        {
            glUnmapNamedBuffer(program.light_ops_atomic_counter_buffer);
            program.light_ops_mapped_pointer = NULL;
        }

        glDeleteBuffers(1, &program.light_ops_atomic_counter_buffer);
    }
    glCreateBuffers(1, &program.light_ops_atomic_counter_buffer);
    // glNamedBufferData(program.light_ops_atomic_counter_buffer, sizeof(u32), NULL, GL_DYNAMIC_DRAW);
    glNamedBufferStorage(program.light_ops_atomic_counter_buffer, sizeof(u32), NULL, GL_MAP_READ_BIT | GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT | GL_DYNAMIC_STORAGE_BIT);
    glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 0, program.light_ops_atomic_counter_buffer);
    
    // Persistantly map atomic counter buffer
    program.light_ops_mapped_pointer = (u32*)glMapNamedBufferRange(program.light_ops_atomic_counter_buffer, 0, sizeof(u32), GL_MAP_READ_BIT | GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT);
    if (!program.light_ops_mapped_pointer)
    {
        printf("Failed to persistantly map atomic counter buffer\n");
        exit(1);
    }
}

void
execute_pbr_draw_call(u32 shader_program, PBRDrawCall* draw_call)
{
    // Set matrix uniforms
    glProgramUniformMatrix4fv(shader_program, PBR_LOC_mvp, 1, GL_FALSE, (f32*)draw_call->mvp);
    glProgramUniformMatrix4fv(shader_program, PBR_LOC_model_view, 1, GL_FALSE, (f32*)draw_call->model_view);
    glProgramUniformMatrix4fv(shader_program, PBR_LOC_normal_matrix, 1, GL_FALSE, (f32*)draw_call->normal_matrix);

    // Set material uniforms
    {
        glProgramUniform4fv(shader_program, PBR_LOC_base_color_factor, 1, (f32*)draw_call->uniforms.base_color_factor);
        glProgramUniform1f(shader_program, PBR_LOC_metallic_factor, draw_call->uniforms.metallic_factor);
        glProgramUniform1f(shader_program, PBR_LOC_roughness_factor, draw_call->uniforms.roughness_factor);
        glProgramUniform3fv(shader_program, PBR_LOC_emissive_factor, 1, (f32*)draw_call->uniforms.emissive_factor);
        glProgramUniform1f(shader_program, PBR_LOC_alpha_mask_cutoff, draw_call->uniforms.alpha_mask_cutoff);
        glProgramUniform1i(shader_program, PBR_LOC_is_normal_mapping_enabled, draw_call->uniforms.is_normal_mapping_enabled);
        glProgramUniform1i(shader_program, PBR_LOC_is_alpha_blending_enabled, draw_call->uniforms.is_alpha_blending_enabled);
    }

    if (draw_call->double_sided)
    {
        glDisable(GL_CULL_FACE);
    }
    else
    {
        glEnable(GL_CULL_FACE);
    }

    // Bind material textures
    glBindTextureUnit(PBR_TEXUNIT_base_color_linear_space, draw_call->texture_ids[PBR_TEXUNIT_base_color_linear_space]);
    glBindTextureUnit(PBR_TEXUNIT_metallic_roughness_texture, draw_call->texture_ids[PBR_TEXUNIT_metallic_roughness_texture]);
    glBindTextureUnit(PBR_TEXUNIT_emissive_texture, draw_call->texture_ids[PBR_TEXUNIT_emissive_texture]);
    glBindTextureUnit(PBR_TEXUNIT_occlusion_texture, draw_call->texture_ids[PBR_TEXUNIT_occlusion_texture]);
    glBindTextureUnit(PBR_TEXUNIT_normal_texture, draw_call->texture_ids[PBR_TEXUNIT_normal_texture]);

    // Bind LTC textures
    glBindTextureUnit(TEXUNIT_LTC1_texture, program.LTC1_texture);
    glBindTextureUnit(TEXUNIT_LTC2_texture, program.LTC2_texture);

    // Bind clustered shading normal cubemap and texture
    if (program.is_clustered_shading_enabled)
    {
        glBindTextureUnit(TEXUNIT_cluster_normals_cubemap, program.cluster_normals_cubemap);
        glBindTextureUnit(TEXUNIT_representative_normals_texture, program.representative_normals_1dtexure);
    }
    
    glBindVertexArray(draw_call->vao);
    
    cgltf_primitive* prim = draw_call->prim;
    if (prim->indices != NULL)
    {
        assert(prim->indices->type == cgltf_type_scalar && "Indices glTF accessor must be use SCALAR types.");

        // Get indices OpenGL component type from cgltf component type
        u32 indices_component_type = gl_component_type_from_cgltf(prim->indices->component_type);
        if (indices_component_type != GL_UNSIGNED_BYTE &&
            indices_component_type != GL_UNSIGNED_SHORT &&
            indices_component_type != GL_UNSIGNED_INT)
        {
            assert(0 && "glDrawElements documentation: Must be one of GL_UNSIGNED_BYTE, GL_UNSIGNED_SHORT, or GL_UNSIGNED_INT.");
        }

        // Find offset that indices start within ebo
        size_t offset = prim->indices->offset + prim->indices->buffer_view->offset;
        glDrawElements(draw_call->primitive_mode, prim->indices->count, indices_component_type, (const void*)offset);
    }
    else
    {
        // The glTF specification lets us get the vertex count using an arbitrary attribute
        assert(prim->attributes_count > 0);
        u32 vertex_count = prim->attributes[0].data->count;
        
        glDrawArrays(draw_call->primitive_mode, 0, vertex_count);
    }
}

PBRDrawCall
build_gltf_primitive_draw_call(Scene* scene, cgltf_mesh* mesh,
    VAO_Range mesh_vao_range, int prim_index,
    mat4 mv_matrix, mat4 mvp_matrix, mat4 normal_matrix)
{
    PBRDrawCall draw_call = { 0 };
    glm_mat4_copy(mv_matrix, draw_call.model_view);
    glm_mat4_copy(mvp_matrix, draw_call.mvp);
    glm_mat4_copy(normal_matrix, draw_call.normal_matrix);

    draw_call.vao = scene->vaos[mesh_vao_range.begin + prim_index];
    VAO_Attributes vao_attributes = scene->vaos_attributes[mesh_vao_range.begin + prim_index];
 
    cgltf_primitive* prim = &mesh->primitives[prim_index];
    draw_call.prim = prim;
    draw_call.primitive_mode = gl_primitive_mode_from_cgltf(prim->type);

    // Bind material
    cgltf_material* material = prim->material;
    if (!material)
    {
        printf("draw_gltf_node: Only supporting glTF files with materials because lazy :). Exiting");
        exit(1);
    }

    draw_call.double_sided = material->double_sided;

    // Can only use normal mapping for vaos with tangents
    draw_call.uniforms.is_normal_mapping_enabled = vao_attributes.has_tangent;
    
    // Set texture uniforms
    cgltf_pbr_metallic_roughness* pbr_mr = &material->pbr_metallic_roughness;
    u32 base_color_id = 0;
    u32 metallic_roughness_id = 0;
    u32 emissive_id = 0;
    u32 occlusion_id = 0;
    u32 normal_id = 0;
    // u32 other texture; etc...

    // Find texture ids for materials textures
    for (u32 tex_i = 0; tex_i < scene->data->textures_count; ++tex_i)
    {
        cgltf_texture* texture = &scene->data->textures[tex_i];

        if (texture == pbr_mr->base_color_texture.texture) base_color_id = tex_i;
        if (texture == pbr_mr->metallic_roughness_texture.texture) metallic_roughness_id = tex_i;
        if (texture == material->emissive_texture.texture) emissive_id = tex_i;
        if (texture == material->occlusion_texture.texture) occlusion_id = tex_i;
        if (texture == material->normal_texture.texture) normal_id = tex_i;
    }

    // Set base color texture
    if (pbr_mr->base_color_texture.texture)
    {
        draw_call.texture_ids[PBR_TEXUNIT_base_color_linear_space] = scene->texture_objects[base_color_id];
    }
    else
    {
        // Fallback to white texture
        draw_call.texture_ids[PBR_TEXUNIT_base_color_linear_space] = scene->white_texture;
    }

    // Set base color factor (cgltf defaults this to white if field not provided in gltf file)
    draw_call.uniforms.base_color_factor[0] = pbr_mr->base_color_factor[0];
    draw_call.uniforms.base_color_factor[1] = pbr_mr->base_color_factor[1];
    draw_call.uniforms.base_color_factor[2] = pbr_mr->base_color_factor[2];
    draw_call.uniforms.base_color_factor[3] = pbr_mr->base_color_factor[3];

    // Set Alpha modes that apply to base color texture
    float alpha_cutoff;
    int is_alpha_blending_enabled;
    if (material->alpha_mode == cgltf_alpha_mode_mask)
    {
        alpha_cutoff = material->alpha_cutoff;
        // is_alpha_blending_enabled = 0;
        is_alpha_blending_enabled = 1;  // TEMPORARY FIX: For early depth tests, masked objects go in transparent pass too
    }
    else if (material->alpha_mode == cgltf_alpha_mode_opaque)
    {
        alpha_cutoff = 0.0f;
        is_alpha_blending_enabled = 0;
    }
    else if (material->alpha_mode == cgltf_alpha_mode_blend)
    {
        alpha_cutoff = 0.0f;
        is_alpha_blending_enabled = 1;
    }
    else
    {
        assert(0 && "Impossible alpha mode unless cgltf.h bugged.");
    }
    draw_call.uniforms.alpha_mask_cutoff = alpha_cutoff;
    draw_call.uniforms.is_alpha_blending_enabled = is_alpha_blending_enabled;

    if (pbr_mr->metallic_roughness_texture.texture)
    {
        // Set metallic roughness texture
        draw_call.texture_ids[PBR_TEXUNIT_metallic_roughness_texture] = scene->texture_objects[metallic_roughness_id];
    }
    else
    {
        // Fallback to default white texture
        draw_call.texture_ids[PBR_TEXUNIT_metallic_roughness_texture] = scene->white_texture;
    }

    // Set metallic and roughness factors
    draw_call.uniforms.metallic_factor = pbr_mr->metallic_factor;
    draw_call.uniforms.roughness_factor = pbr_mr->roughness_factor;

    if (material->emissive_texture.texture)
    {
        // Set emissive texture
        draw_call.texture_ids[PBR_TEXUNIT_emissive_texture] = scene->texture_objects[emissive_id];
    }
    else
    {
        // Fallback to default white texture
        draw_call.texture_ids[PBR_TEXUNIT_emissive_texture] = scene->white_texture;
    }

    // Set emissive factor
    draw_call.uniforms.emissive_factor[0] = material->emissive_factor[0];
    draw_call.uniforms.emissive_factor[1] = material->emissive_factor[1];
    draw_call.uniforms.emissive_factor[2] = material->emissive_factor[2];

    if (material->occlusion_texture.texture)
    {
        // Set occlusion texture
        draw_call.texture_ids[PBR_TEXUNIT_occlusion_texture] = scene->texture_objects[occlusion_id];
    }
    else
    {
        // Fallback to default white texture
        draw_call.texture_ids[PBR_TEXUNIT_occlusion_texture] = scene->white_texture;
    }

    if (material->normal_texture.texture)
    {
        // Set normal texture
        draw_call.texture_ids[PBR_TEXUNIT_normal_texture] =  scene->texture_objects[normal_id];
    }
    else
    {
        // Fallback to flat normal map texture
        draw_call.texture_ids[PBR_TEXUNIT_normal_texture] = scene->flat_normal_texture;
    }

    return draw_call;
}

void
add_gltf_node_draw_calls(Scene* scene, FreeCamera* camera, u32 opaque_program, cgltf_node* node, mat4 parent_matrix, DynamicArray* opaque_draw_calls, DynamicArray* transparent_draw_calls)
{
    // Compute model matrix = parent_matrix * node's matrix   
    mat4 model = GLM_MAT4_IDENTITY_INIT;
    {
#if 0  // Turns out cgltf provides an implementation to get the node world transform
        cgltf_node_transform_world(node, (float*)model);
#else
        // Node transform either in matrix format matrix=T*R*S or T,R,S seperately (translation vector, rotation quaternion, scale vector)

        if (node->has_matrix)
        {
            mat4 node_matrix = {
                { node->matrix[0],  node->matrix[1],  node->matrix[2],  node->matrix[3] },
                { node->matrix[4],  node->matrix[5],  node->matrix[6],  node->matrix[7] },
                { node->matrix[8],  node->matrix[9],  node->matrix[10], node->matrix[11] },
                { node->matrix[12], node->matrix[13], node->matrix[14], node->matrix[15] }
            };

            glm_mat4_mul(parent_matrix, node_matrix, model);
        }
        else
        {
            mat4 node_matrix = GLM_MAT4_IDENTITY_INIT;

            if (node->has_translation)
            {
                vec3 translation_vector = { node->translation[0], node->translation[1], node->translation[2] };
                glm_translate(node_matrix, translation_vector);
            }
            
            if (node->has_rotation)
            {
                // printf("rot: %f, %f, %f, %f\n", node->rotation[0], node->rotation[1], node->rotation[2], node->rotation[3]);
                versor rotation_quaternion = { node->rotation[0], node->rotation[1], node->rotation[2], node->rotation[3] };
                glm_quat_rotate(node_matrix, rotation_quaternion, node_matrix);
            }

            if (node->has_scale)
            {
                vec3 scale_vector = { node->scale[0], node->scale[1], node->scale[2] };
                glm_scale(node_matrix, scale_vector);
            }

            glm_mat4_mul(parent_matrix, node_matrix, model);
        }
#endif
    }
    
    if (node->mesh)
    {
        // Compute and upload our vertex shader matrices
        mat4 mv_matrix; glm_mat4_mul(camera->view_matrix, model, mv_matrix);
        mat4 mvp_matrix; glm_mat4_mul(camera->camera_matrix, model, mvp_matrix);
        mat4 normal_matrix; glm_mat4_inv(mv_matrix, normal_matrix); glm_mat4_transpose(normal_matrix);

        // glProgramUniformMatrix4fv(opaque_program, PBR_LOC_mvp, 1, GL_FALSE, (f32*)mvp_matrix);
        // glProgramUniformMatrix4fv(opaque_program, PBR_LOC_model_view, 1, GL_FALSE, (f32*)mv_matrix);
        // glProgramUniformMatrix4fv(opaque_program, PBR_LOC_normal_matrix, 1, GL_FALSE, (f32*)normal_matrix);

        // Find node's mesh and get its index
        cgltf_mesh* mesh = node->mesh;  // e.g. nodes->mesh = &scene->data->meshes[2];
        s64 mesh_index = mesh - scene->data->meshes;
        assert(0 <= mesh_index && mesh_index < (s64)scene->data->meshes_count);

        VAO_Range mesh_vao_range = scene->vao_ranges[mesh_index];

        // Iterate over the mesh's primitives and draw each primitives VAO
        for (u32 prim_i = 0; prim_i < mesh->primitives_count; ++prim_i)
        {
            PBRDrawCall draw_call = build_gltf_primitive_draw_call(scene, mesh, mesh_vao_range, prim_i, mv_matrix, mvp_matrix, normal_matrix);

            if (draw_call.uniforms.is_alpha_blending_enabled)
            {
                // Add to transparent draw calls
                push_element_copy(transparent_draw_calls, sizeof(PBRDrawCall), &draw_call);
            }
            else
            {
                // Add to opaque draw calls
                push_element_copy(opaque_draw_calls, sizeof(PBRDrawCall), &draw_call);
            }
        }
    }

    // Draw children nodes
    for (u32 i = 0; i < node->children_count; ++i)
    {
        cgltf_node* child = node->children[i];
        add_gltf_node_draw_calls(scene, camera, opaque_program, child, model, opaque_draw_calls, transparent_draw_calls);
    }
}

// int
// compare_draw_call_depths(const void* draw_call_a, const void* draw_call_b)
// {
//     const PBRDrawCall* a = (PBRDrawCall*)draw_call_a;
//     const PBRDrawCall* b = (PBRDrawCall*)draw_call_b;

//     float a_depth = a->model_view[3][2];
//     float b_depth = b->model_view[3][2];

//     if (a_depth < b_depth)
//     {
//         return 1;
//     }
//     else if (a_depth > b_depth)
//     {
//         return -1;
//     }
//     else
//     {
//         return 0;
//     }
// }

u32
load_ltc_matrix_texture(const float* matrix_table)
{
    u32 tex = 0;
    glCreateTextures(GL_TEXTURE_2D, 1, &tex);
    glTextureStorage2D(tex, 1, GL_RGBA32F, 64, 64);
    glTextureSubImage2D(tex, 0, 0, 0, 64, 64, GL_RGBA, GL_FLOAT, matrix_table);
    
    // Paper uses bilinear filtering of this texture to interpolate matrices....
    glTextureParameteri(tex, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTextureParameteri(tex, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTextureParameteri(tex, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(tex, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    return tex;
}

void
render_area_lights(int light_count, AreaLight* arealights)
{
    /* Renders the area light's polygon with a simple polygon shader (polygon.vert/frag) */

    // Create a temporary VAO/VBO each frame
    GLuint vao, vbo, ebo;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    // Create ebo for star since its concave so we can't use GL_TRIANGLE_FAN
    glGenBuffers(1, &ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(star_indices), star_indices, GL_STATIC_DRAW);


    // Build Vertex Data
    int total_vertices = 0;
    for (int i = 0; i < light_count; i++)
        total_vertices += arealights[i].n; // Total vertices needed

    float* vertex_data = malloc(total_vertices * sizeof(vec3));
    float* color_data = malloc(total_vertices * sizeof(vec3));
    
    int index = 0;
    for (int i = 0; i < light_count; i++)
    {
        AreaLight* al = &arealights[i];

        for (int j = 0; j < al->n; j++)
        {
            // glm_vec3_copy((vec3){ al->points_worldspace[j][0], al->points_worldspace[j][1], al->points_worldspace[j][2] },
                // &vertex_data[index * 3]);
            vertex_data[index * 3 + 0] = al->points_worldspace[j][0];
            vertex_data[index * 3 + 1] = al->points_worldspace[j][1];
            vertex_data[index * 3 + 2] = al->points_worldspace[j][2];
            
            // glm_vec3_copy((vec3){ al->color_rgb_intensity_a[0], al->color_rgb_intensity_a[1], al->color_rgb_intensity_a[2] },
            //     &color_data[index * 3]);
            color_data[index * 3 + 0] = al->color_rgb_intensity_a[0];
            color_data[index * 3 + 1] = al->color_rgb_intensity_a[1];
            color_data[index * 3 + 2] = al->color_rgb_intensity_a[2];

            index++;
        }
    }

    // Upload Data
    glBufferData(GL_ARRAY_BUFFER, total_vertices * (sizeof(vec3) * 2), NULL, GL_DYNAMIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, total_vertices * sizeof(vec3), vertex_data);
    glBufferSubData(GL_ARRAY_BUFFER, total_vertices * sizeof(vec3), total_vertices * sizeof(vec3), color_data);

    free(vertex_data);
    free(color_data);

    // Setup Vertex Attributes
    glEnableVertexAttribArray(0); // Position
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vec3), (void*)0);
    
    glEnableVertexAttribArray(1); // Color
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(vec3), (void*)(total_vertices * sizeof(vec3)));

    // Render
    glUseProgram(program.shader_area_light_polygons);
    glProgramUniformMatrix4fv(program.shader_area_light_polygons, 0, 1, GL_FALSE, (f32*)program.cam.camera_matrix);

    // glPointSize(24.0f);
    // glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);
    int offset = 0;
    for (int i = 0; i < light_count; i++)
    {
        if (arealights[i].n == 10)
        {
            glDrawElementsBaseVertex(GL_TRIANGLES, sizeof(star_indices) / sizeof(star_indices[0]), GL_UNSIGNED_INT, 0, offset);
        }
        else
        {
            glDrawArrays(GL_TRIANGLE_FAN, offset, arealights[i].n);
        }
        offset += arealights[i].n;
    }
    // glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    // Cleanup
    glDeleteBuffers(1, &vbo);
    glDeleteBuffers(1, &ebo);
    glDeleteVertexArrays(1, &vao);
}

void
draw_gltf_scene(Scene* scene)
{
    FreeCamera* camera = &program.cam;
    u32 shader_program = program.shader_pbr_opaque;
    u32 compute_clusters_shader = program.shader_compute_clusters;
    u32 light_assignment_shader = program.shader_light_assignment;
    b32 enable_clustered_shading = program.is_clustered_shading_enabled;

    // OLD UNNECESSARY: Make sure SSBOs are aren't unbound by thirdparty GUI library
    // glBindBufferBase(GL_SHADER_STORAGE_BUFFER, GLOBAL_SSBO_INDEX_POINTLIGHTS, program.point_light_ssbo);
    // glBindBufferBase(GL_SHADER_STORAGE_BUFFER, GLOBAL_SSBO_INDEX_AREALIGHTS, program.area_light_ssbo);
    // glBindBufferBase(GL_SHADER_STORAGE_BUFFER, GLOBAL_SSBO_INDEX_CLUSTERGRID, program.cluster_grid_ssbo);

    // Reset light op counter to 0
    // u32 zero = 0;
    // glNamedBufferSubData(program.light_ops_atomic_counter_buffer, 0, sizeof(u32), &zero);
    *program.light_ops_mapped_pointer = 0;

    u32 num_point_lights = array_length(&program.point_lights, sizeof(PointLight));
    u32 num_area_lights = array_length(&program.area_lights, sizeof(AreaLight));

    // Upload lights in viewspace from program.point_lights to the light SSBOs
    {
        // Resize point and area light SSBOs

        if (num_point_lights > program.point_light_ssbo_max)  // For verification purposes resize the ssbo every time the amount of lights changes
        {
            program.point_light_ssbo_max = max(1, num_point_lights);  // Increase buffer by increments of 50 point lights
            size_t new_size = sizeof(PointLight) * program.point_light_ssbo_max;
            glNamedBufferData(program.point_light_ssbo, new_size, NULL, GL_DYNAMIC_DRAW);
            
            // Old code where i recreated the buffer and then had to rebind buffer base
            // glDeleteBuffers(1, &scene->point_light_ssbo);
            // glCreateBuffers(1, &scene->point_light_ssbo);
            // glNamedBufferData(scene->point_light_ssbo, new_size, NULL, GL_DYNAMIC_DRAW);
            // glBindBufferBase(GL_SHADER_STORAGE_BUFFER, GLOBAL_SSBO_INDEX_POINTLIGHTS, scene->point_light_ssbo);
        }

        if (num_area_lights > program.area_light_ssbo_max)
        {
            program.area_light_ssbo_max = max(1, num_area_lights);
            size_t new_size = sizeof(AreaLight) * program.area_light_ssbo_max;
            glNamedBufferData(program.area_light_ssbo, new_size, NULL, GL_DYNAMIC_DRAW);
        }

        // Update point lights SSBO:
        f32* mapped_pl_ssbo = (float*)glMapNamedBuffer(program.point_light_ssbo, GL_WRITE_ONLY);
        for (u32 point_id = 0; point_id < num_point_lights; ++point_id)
        {
            PointLight* point_light = get_element(&program.point_lights, sizeof(PointLight), point_id);
            
            // Transform point light position to view space
            vec4 viewpos = { point_light->position[0], point_light->position[1], point_light->position[2], 1.0f };
            glm_mat4_mulv(camera->view_matrix, viewpos, viewpos);

            // Compute range for point light
            float point_light_range = calculate_point_light_range(
                point_light->intensity, scene->minimum_perceivable_intensity,
                scene->attenuation_quadratic,
                scene->attenuation_linear,
                scene->attenuation_constant
            );
            // printf("range: %f\n", point_light_range);

            float* mapped_pl = &mapped_pl_ssbo[point_id * sizeof(PointLight) / sizeof(f32)];

            // Set mapped vec4 position_xyz_range_w
            mapped_pl[0] = viewpos[0];
            mapped_pl[1] = viewpos[1];
            mapped_pl[2] = viewpos[2];
            mapped_pl[3] = point_light_range;

            // Set mapped vec4 color_rgb_intensity_a
            mapped_pl[4] = point_light->color[0];
            mapped_pl[5] = point_light->color[1];
            mapped_pl[6] = point_light->color[2];
            mapped_pl[7] = point_light->intensity;
        }
        glUnmapNamedBuffer(program.point_light_ssbo);
        
        // Update area lights SSBO:
        f32* mapped_arealight_ssbo = (float*)glMapNamedBuffer(program.area_light_ssbo, GL_WRITE_ONLY);
        for (u32 area_id = 0; area_id < num_area_lights; ++area_id)
        {
            AreaLight* area_light = get_element(&program.area_lights, sizeof(AreaLight), area_id);
            
            // Transform area light polygon from world to view space
            vec4 points_viewspace[MAX_UNCLIPPED_NGON];
            for (int vertex = 0; vertex < MAX_UNCLIPPED_NGON; ++vertex)
            {
                glm_mat4_mulv(camera->view_matrix, area_light->points_worldspace[vertex], points_viewspace[vertex]);
                // glm_vec4_copy(area_light->points_worldspace[vertex], points_viewspace[vertex]);
            }
            
            // Compute bounding box and sphere:
            vec4 aabb_min;
            vec4 aabb_max;
            vec4 sphere_of_influence;
            {
                // Find average of points
                vec3 centroid = { 0.0f, 0.0f, 0.0f };
                for (int i = 0; i < area_light->n; ++i)
                {
                    glm_vec3_add(centroid, points_viewspace[i], centroid);
                }
                glm_vec3_scale(centroid, 1.0f / (float)area_light->n, centroid);
                
                // Find point with max distance from centroid
                float geo_radius = 0.0f;  
                for (int i = 0; i < area_light->n; ++i)
                {
                    float distance = glm_vec3_distance(centroid, points_viewspace[i]);
                    if (distance > geo_radius)
                    {
                        geo_radius = distance;
                    }
                }

                float area = polygon_area(area_light);
                float influence_radius = calculate_area_light_influence_radius(area_light, area, scene->minimum_perceivable_intensity, scene->param_roughness);
                
                sphere_of_influence[0] = centroid[0];
                sphere_of_influence[1] = centroid[1];
                sphere_of_influence[2] = centroid[2];
                sphere_of_influence[3] = geo_radius + influence_radius;
                
                // Also use AABB for early rejection in light-cluster assignment
                glm_vec4_copy(points_viewspace[0], aabb_min);
                glm_vec4_copy(points_viewspace[0], aabb_max);
                for (int i = 0; i < area_light->n; ++i)
                {
                    if (aabb_min[0] > points_viewspace[i][0])
                        aabb_min[0] = points_viewspace[i][0];
                    if (aabb_max[0] < points_viewspace[i][0])
                        aabb_max[0] = points_viewspace[i][0];
                    
                    if (aabb_min[1] > points_viewspace[i][1])
                        aabb_min[1] = points_viewspace[i][1];
                    if (aabb_max[1] < points_viewspace[i][1])
                        aabb_max[1] = points_viewspace[i][1];

                    if (aabb_min[2] > points_viewspace[i][2])
                        aabb_min[2] = points_viewspace[i][2];
                    if (aabb_max[2] < points_viewspace[i][2])
                        aabb_max[2] = points_viewspace[i][2];
                }

                vec4 influence_vec = { influence_radius, influence_radius, influence_radius, 0.0f };
                glm_vec4_sub(aabb_min, influence_vec, aabb_min);
                glm_vec4_add(aabb_max, influence_vec, aabb_max);
            }

            float* mapped_arealight = &mapped_arealight_ssbo[area_id * sizeof(AreaLight) / sizeof(f32)];

            // Set mapped color and intensity
            mapped_arealight[0] = area_light->color_rgb_intensity_a[0];
            mapped_arealight[1] = area_light->color_rgb_intensity_a[1];
            mapped_arealight[2] = area_light->color_rgb_intensity_a[2];
            mapped_arealight[3] = area_light->color_rgb_intensity_a[3];
            
            ((int*)mapped_arealight)[4] = area_light->n;
            ((int*)mapped_arealight)[5] = area_light->is_double_sided;
            mapped_arealight[6] = area_light->_packing0;
            mapped_arealight[7] = area_light->_packing1;
            
            // Set mapped cluster parameters
            mapped_arealight[8]  = aabb_min[0];
            mapped_arealight[9]  = aabb_min[1];
            mapped_arealight[10] = aabb_min[2];
            mapped_arealight[11] = aabb_min[3];

            mapped_arealight[12] = aabb_max[0];
            mapped_arealight[13] = aabb_max[1];
            mapped_arealight[14] = aabb_max[2];
            mapped_arealight[15] = aabb_max[3];

            mapped_arealight[16] = sphere_of_influence[0];
            mapped_arealight[17] = sphere_of_influence[1];
            mapped_arealight[18] = sphere_of_influence[2];
            mapped_arealight[19] = sphere_of_influence[3];

            // Set mapped area light points
            for (int vertex = 0; vertex < MAX_UNCLIPPED_NGON; ++vertex)
            {
                mapped_arealight[20 + vertex*4 + 0] = points_viewspace[vertex][0];
                mapped_arealight[20 + vertex*4 + 1] = points_viewspace[vertex][1];
                mapped_arealight[20 + vertex*4 + 2] = points_viewspace[vertex][2];
                mapped_arealight[20 + vertex*4 + 3] = points_viewspace[vertex][3];
            }
        }
        glUnmapNamedBuffer(program.area_light_ssbo);
    }

    if (enable_clustered_shading)
    {
        // Ensure query has been used at least once before fetching compute time
        if (program.compute_time_query && program.frame_counter > 0)
        {
            int available = 0;
            glGetQueryObjectiv(program.compute_time_query, GL_QUERY_RESULT_AVAILABLE, &available);
            if (available)
            {
                glGetQueryObjectui64v(program.compute_time_query, GL_QUERY_RESULT, &program.compute_time_last_frame);
            }
        }

        glBeginQuery(GL_TIME_ELAPSED, program.compute_time_query);

        // TODO: Also implement froxel clusters for comparison?

        // Compute viewspace cluster AABBs with a compute shader
        glUseProgram(compute_clusters_shader);  // voxel_clusters_viewspace.comp

        mat4 inv_proj;
        glm_mat4_inv(camera->projection_matrix, inv_proj);
        glProgramUniform1f(compute_clusters_shader, 0, camera->near_plane);
        glProgramUniform1f(compute_clusters_shader, 1, camera->far_plane);
        glProgramUniformMatrix4fv(compute_clusters_shader, 2, 1, GL_FALSE, (f32*)inv_proj);
        glProgramUniform4ui(compute_clusters_shader, 3, CLUSTER_GRID_SIZE_X, CLUSTER_GRID_SIZE_Y, CLUSTER_GRID_SIZE_Z, CLUSTER_NORMALS_COUNT);
        glProgramUniform2ui(compute_clusters_shader, 4, camera->width, camera->height);

        #define EFFICIENT_WORKGROUPS
        #ifdef EFFICIENT_WORKGROUPS
            glDispatchCompute(1, 1, CLUSTER_GRID_SIZE_Z * CLUSTER_NORMALS_COUNT);
        #else
            glDispatchCompute(CLUSTER_GRID_SIZE_X, CLUSTER_GRID_SIZE_Y, CLUSTER_GRID_SIZE_Z * CLUSTER_NORMALS_COUNT);
        #endif

        // Make sure the writes to the cluster SSBO happen before the next shader
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        // glMemoryBarrier(GL_ALL_BARRIER_BITS);
        // glFinish();

        // Assign lights to clusters with a second compute shader
        glUseProgram(light_assignment_shader);  // lights_to_clusters.comp

        // glProgramUniformMatrix4fv(light_assignment_shader, 0, 1, GL_FALSE, (f32*)camera->view_matrix);
        glProgramUniform1ui(light_assignment_shader, 1, num_point_lights);
        glProgramUniform1ui(light_assignment_shader, 2, num_area_lights);
        glProgramUniform1f(light_assignment_shader, 3, scene->param_roughness);
        glProgramUniform1f(light_assignment_shader, 4, scene->param_min_intensity);
        glProgramUniform1f(light_assignment_shader, 5, scene->param_intensity_saturation);

#ifdef INTEGRATED_GPU
        const u32 LIGHT_ASSIGNMENT_LOCAL_SIZE = 512;
#else
        const u32 LIGHT_ASSIGNMENT_LOCAL_SIZE = 64;
#endif
        const int dispatched_workgroups = NUM_CLUSTERS / LIGHT_ASSIGNMENT_LOCAL_SIZE;
        assert(NUM_CLUSTERS % LIGHT_ASSIGNMENT_LOCAL_SIZE == 0);

        glDispatchCompute(dispatched_workgroups, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        // glMemoryBarrier(GL_ALL_BARRIER_BITS);
        // glFinish();

        glEndQuery(GL_TIME_ELAPSED);
    }
    

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(shader_program);

    if (enable_clustered_shading)
    {
        // Clustered shading uniform params
        glProgramUniform1f(shader_program, PBR_LOC_near, camera->near_plane);
        glProgramUniform1f(shader_program, PBR_LOC_far, camera->far_plane);
        glProgramUniform4ui(shader_program, PBR_LOC_grid_size, CLUSTER_GRID_SIZE_X, CLUSTER_GRID_SIZE_Y, CLUSTER_GRID_SIZE_Z, CLUSTER_NORMALS_COUNT);
        glProgramUniform2ui(shader_program, PBR_LOC_screen_dimensions, camera->width, camera->height);
    }
    else
    {
        glProgramUniform1ui(shader_program, PBR_LOC_num_point_lights, num_point_lights);
        glProgramUniform1ui(shader_program, PBR_LOC_num_area_lights, num_area_lights);
    }

    // Compute and upload light data
    {
        // Upload Directional Light in View Space
        {
            // Transform sun direction to view space
            vec3 sun_direction_viewspace;
            vec4 v = { scene->sun_direction[0], scene->sun_direction[1], scene->sun_direction[2], 0.0f };
            glm_mat4_mulv(camera->view_matrix, v, v);
            glm_vec3_normalize(v);
            glm_vec3_copy(v, sun_direction_viewspace);

            glProgramUniform3fv(shader_program, PBR_LOC_sun_direction_viewspace, 1, (f32*)sun_direction_viewspace);
            glProgramUniform1f(shader_program, PBR_LOC_sun_intensity, scene->sun_intensity);
            glProgramUniform3fv(shader_program, PBR_LOC_sun_color, 1, (f32*)scene->sun_color);
        }

        // Upload Attenuation parameters
        {
            glProgramUniform1f(shader_program, PBR_LOC_constant_attenuation, scene->attenuation_constant);
            glProgramUniform1f(shader_program, PBR_LOC_linear_attenuation, scene->attenuation_linear);
            glProgramUniform1f(shader_program, PBR_LOC_quadratic_attenuation, scene->attenuation_quadratic);
        }
    }

    // Setup different draw call arrays based on alpha modes
    // Using dynamic array even though the gltf file is constant in order to scale for adding object into the scene dynamically
    DynamicArray opaque_draw_calls = create_array(scene->total_opaque_primitives * sizeof(PBRDrawCall));
    DynamicArray transparent_draw_calls = create_array(scene->total_transparent_primitives * sizeof(PBRDrawCall));

    // Draw the scene specified by gltf file
    if (scene->data->scene)
    {
        // Add draw calls to either opaque or transparent
        for (u32 i = 0; i < scene->data->nodes_count; ++i)
        {
            cgltf_node* node = &scene->data->nodes[i];
            add_gltf_node_draw_calls(scene, camera, shader_program, node, GLM_MAT4_IDENTITY, &opaque_draw_calls, &transparent_draw_calls);
        }
    }

    // Opaque render pass
    u32 num_opaques = array_length(&opaque_draw_calls, sizeof(PBRDrawCall));
    for (u32 opaque_id = 0; opaque_id < num_opaques; ++opaque_id)
    {
        PBRDrawCall* draw_call = get_element(&opaque_draw_calls, sizeof(PBRDrawCall), opaque_id);
        execute_pbr_draw_call(shader_program, draw_call);
    }
    
    // Transparent render pass (includes alphamasked objects for now as well)
    u32 num_transparents = array_length(&transparent_draw_calls, sizeof(PBRDrawCall));
    // OLD: qsort won't work for suntemple due to seperate trees being stored in one primitive so they are in the same draw call, order independant method required
    // qsort(transparent_draw_calls.data_buffer, transparent_draw_calls.used_size / sizeof(PBRDrawCall), sizeof(PBRDrawCall), compare_draw_call_depths);
    for (u32 transparent_id = 0; transparent_id < num_transparents; ++transparent_id)
    {
        PBRDrawCall* draw_call = get_element(&transparent_draw_calls, sizeof(PBRDrawCall), transparent_id);
        execute_pbr_draw_call(shader_program, draw_call);
    }
    
    // Render area lights
    // glEnable(GL_CULL_FACE);  // Must explicitly reenable this in case a double sided item was just rendered.
    glDisable(GL_CULL_FACE);
    render_area_lights(num_area_lights, program.area_lights.data_buffer);

    free_array(&opaque_draw_calls);
    free_array(&transparent_draw_calls);

    // Check number of light ops
    // glGetNamedBufferSubData(program.light_ops_atomic_counter_buffer, 0, sizeof(u32), &program.last_light_ops_value);
    program.last_light_ops_value = *program.light_ops_mapped_pointer;
}

void
free_scene(Scene scene)
{
    glDeleteBuffers(scene.data->buffers_count, scene.buffer_objects);
    glDeleteTextures(scene.data->textures_count, scene.texture_objects);
    glDeleteTextures(1, &scene.white_texture);
    glDeleteTextures(1, &scene.flat_normal_texture);
    glDeleteVertexArrays(scene.vaos_count, scene.vaos);
    glDeleteBuffers(1, &program.point_light_ssbo);
    glDeleteBuffers(1, &program.cluster_grid_ssbo);

    if (scene.data) cgltf_free(scene.data);
    if (scene.buffer_objects) free(scene.buffer_objects);
    if (scene.texture_objects) free(scene.texture_objects);
    if (scene.vaos) free(scene.vaos);
    if (scene.vaos_attributes) free(scene.vaos_attributes);
    if (scene.vao_ranges) free(scene.vao_ranges);
    free_array(&program.point_lights);
}

void
load_test_scene(int scene_id, Scene* out_loaded_scene)
{
    u32 num_point_lights;
    vec3* point_light_positions;
    if (scene_id == 0)
    {
        *out_loaded_scene = load_gltf_scene("data/sponza-glTF/Sponza.gltf");

        num_point_lights = sizeof(sponza_pointlight_positions) / sizeof(vec3);
        point_light_positions = sponza_pointlight_positions;

        // Give sponza lots of attenuation to fill the small space with *cullable* point lights
        out_loaded_scene->attenuation_constant = 1.0f;
        out_loaded_scene->attenuation_linear    = 50.0f;//= 8.0f;
        out_loaded_scene->attenuation_quadratic = 80.0f;//= 10.0f;

        // Sponza isn't very shiny
        out_loaded_scene->param_roughness = 1.0f;
        out_loaded_scene->param_min_intensity = 0.01f;
        out_loaded_scene->param_intensity_saturation = 100.0f;
    }
    else if (scene_id == 1)
    {
        // *out_loaded_scene = load_gltf_scene("data/suntemple/suntemplegltf.gltf");
        *out_loaded_scene = load_gltf_scene("data/suntemple/mirroredsuntemple/mirroredsuntemple.gltf");

        num_point_lights = sizeof(suntemple_pointlight_positions) / sizeof(vec3);
        point_light_positions = suntemple_pointlight_positions;

        // Give Suntemple more point light attenuation
        out_loaded_scene->attenuation_constant  = 1.0f;
        out_loaded_scene->attenuation_linear    = 30.0f;//14.0f;//= 8.0f;
        out_loaded_scene->attenuation_quadratic = 50.0f;//25.0f;//= 10.0f;

        // Suntemple has some shinies but not like glossy
        out_loaded_scene->param_roughness = 1.0f;
        out_loaded_scene->param_min_intensity = 0.01f;
        out_loaded_scene->param_intensity_saturation = 10.0f;
    }
    else if (scene_id == 2)
    {
        *out_loaded_scene = load_gltf_scene("data/lost-empire/lostempireblenderexport.gltf");

        num_point_lights = sizeof(lostempire_pointlight_positions) / sizeof(vec3);
        point_light_positions = lostempire_pointlight_positions;

        // Give lost empire low point light attenuation
        out_loaded_scene->attenuation_constant = 1.0f;
        out_loaded_scene->attenuation_linear = 2.5f;
        out_loaded_scene->attenuation_quadratic = 5.0f;

        // Lost empire is similar to suntemple roughness wise
        out_loaded_scene->param_roughness = 1.0f;
        out_loaded_scene->param_min_intensity = 0.01f;
        out_loaded_scene->param_intensity_saturation = 10.0f;
    }
    else if (scene_id == 3)
    {
        // All other scenes:
        // NOTE: Currently not supporting image buffer views which is why I'm not using .glb files.
        // *out_loaded_scene = load_gltf_scene("data/Xbox - Halo 2 - Coagulation/Coagulation/glTF/untitled.gltf");
        // *out_loaded_scene = load_gltf_scene("data/Wii U - Mario Kart 8 - Wii Warios Gold Mine/glTF/untitled.gltf");
        // *out_loaded_scene = load_gltf_scene("data/uploads_files_3363978_BackStreet2/Building/gltf/exportfromfbx.gltf");
        *out_loaded_scene = load_gltf_scene("data/blenderflattest/myflattest.gltf");

        // Working scenes:
        // *out_loaded_scene = load_gltf_scene("data/sponza-glTF/Sponza.gltf");
        // *out_loaded_scene = load_gltf_scene("data/suntemple/suntemplegltf.gltf");
        // *out_loaded_scene = load_gltf_scene("data/lost-empire/lostempireblenderexport.gltf");

        // Other working scenes but not worth using for project
        // *out_loaded_scene = load_gltf_scene("data/damagedHemlet-glTF/DamagedHelmet.gltf");
        // *out_loaded_scene = load_gltf_scene("data/scifi-helmet-glTF/SciFiHelmet.gltf");
        // *out_loaded_scene = load_gltf_scene("data/AlphaBlendModeTestglTF/AlphaBlendModeTest.gltf");
        // *out_loaded_scene = load_gltf_scene("data/blendersponza/sponzaexport.gltf");  // <--Normal mapping gone in export?
        // *out_loaded_scene = load_gltf_scene("data/blendercube/cube.gltf");  // <- random blender project to test instancing
        // *out_loaded_scene = load_gltf_scene("data/3DS - The Legend of Zelda Majoras Mask 3D - Snowhead Temple/exported2/snowhead-shrunk.gltf");  // <- Obviously can't include this
        // *out_loaded_scene = load_gltf_scene("data/EmissiveStrengthTest-glTF/EmissiveStrengthTest.gltf");
        // *out_loaded_scene = load_gltf_scene("data/AntiqueCameraglTF/AntiqueCamera.gltf");

        // These intel sponza files are humungous:
        // *out_loaded_scene = load_gltf_scene("data/main1_sponza/NewSponza_Main_glTF_003.gltf");
        // *out_loaded_scene = load_gltf_scene("data/pkg_a_curtains/NewSponza_Curtains_glTF.gltf");
        // *out_loaded_scene = load_gltf_scene("data/pkg_c1_trees/NewSponza_CypressTree_glTF.gltf");


        // Not working scenes
        // *out_loaded_scene = load_gltf_scene("data/simple-instancing-glTF/SimpleInstancing.gltf");  // Don't support no materials
        // *out_loaded_scene = load_gltf_scene("data/fox-glTF/Fox.gltf");  // Don't support vertex colors

        num_point_lights = 0;
        point_light_positions = NULL;

        // Give massive point light attenuation
        out_loaded_scene->attenuation_constant = 1.0f;
        out_loaded_scene->attenuation_linear = 0.5f;
        out_loaded_scene->attenuation_quadratic = 0.2f;

        // Flat test has very glossy aspects
        out_loaded_scene->param_roughness = 0.33f;
        out_loaded_scene->param_min_intensity = 0.01f;
        out_loaded_scene->param_intensity_saturation = 0.5f;

        // Camera settings for reproducing exact test scene
        glm_vec3_copy((vec3){ -20.200048f, 16.906214f, -126.027779f }, program.cam.pos);
        program.cam.pitch = 0.299047f;
        program.cam.yaw = 2.456063f;
    }
    else
    {
        assert(0 && "Invalid test scene id");
    }

    // Init point lights array
    free_array(&program.point_lights);
    program.point_lights = create_array(num_point_lights * sizeof(PointLight));

    // Init area lights array
    free_array(&program.area_lights);
    program.area_lights = create_array(1 * sizeof(AreaLight));

    // Seed RNG so they spawn the same way
    srand(12345);

    // Fill point lights array with initial point lights
    PointLight* lights = push_size(&program.point_lights, sizeof(PointLight), num_point_lights);
    for (u32 i = 0; i < num_point_lights; ++i)
    {
        PointLight* light = &lights[i];

        memset(light, 0, sizeof(PointLight));
        glm_vec3_copy(point_light_positions[i], light->position);
        glm_vec3_copy((vec3){ randomf(), randomf(), randomf() }, light->color);
        light->intensity = rng_rangef(3.0f, 8.0f);
    }

    // Spawn some area lights
    if (scene_id == 1)
    {
        AreaLight al;
        // al = make_area_light((vec3){5.270933,-4.543071,-53.789848}, (vec3){-0.694041,-0.684534,0.222981}, 0, 3, 0.2f, 1.0f, 1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
        al = make_area_light((vec3){5.410935,-2.272083,-48.796303}, (vec3){0.749683,-0.656189,-0.085968}, 0, 4, 0.5f, 25.0f, 2.0f, 3.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
        al = make_area_light((vec3){-2.931523,3.543194,-46.665058}, (vec3){-0.569861,0.701331,0.428245}, 0, 3, 0.6f , 20.0f, 2.0f, 3.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
        
        const int many_light_suntemple_test = 1;
        if (many_light_suntemple_test)
        {
            // Hallway:
            al = make_area_light((vec3){-3.227366,-1.989864,-84.067825}, (vec3){0.967123,-0.254281,-0.003849}, 0, 3,  0.8f, 5.0f, 2.0f, 2.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){3.632351,-1.989864,-84.619041}, (vec3){-0.964972,-0.256029,0.057253}, 0, 3,   0.7f, 5.0f, 2.0f, 2.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-0.171629,-4.844074,-84.749199}, (vec3){-0.009824,-0.981175,-0.192872}, 0, 4, 0.6f, 5.0f, 2.0f, 2.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){2.027821,2.714062,-76.314697}, (vec3){-0.252817,0.966831,-0.036355}, 0, 3,    0.5f, 5.0f, 2.0f, 2.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-2.644127,2.714062,-76.601562}, (vec3){0.545110,0.838352,0.004645}, 0, 3,     0.4f, 5.0f, 2.0f, 2.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-2.806165,0.988074,-67.750755}, (vec3){0.187510,0.981744,-0.031910}, 0, 3,    0.3f, 5.0f, 2.0f, 2.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){3.680951,0.330242,-67.938171}, (vec3){-0.041557,0.984393,-0.171007}, 0, 3,    0.2f, 5.0f, 2.0f, 2.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-0.045660,-3.448810,-57.920753}, (vec3){-0.016721,0.048483,0.998684}, 0, 5,   0.1f, 5.0f, 2.0f, 2.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            // New ones:
            al = make_area_light((vec3){1.864715,0.563292,-39.060966}, (vec3){0.433046,-0.067180,-0.898865}, 0, 5, -0.3f, 8.0f, 1.0f, 1.5f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){4.539085,1.353073,-9.784513}, (vec3){0.477520,0.076000,0.875328}, 0, 4, -1.0f, 20.0f, 1.0f, 3.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-5.330230,9.963239,-6.978376}, (vec3){-0.250825,-0.037807,0.967294}, 0, 4, -1.0f, 20.0f, 4.0f, 1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            
            // Outside lights scattered everywhere (Randomised each scene run)
            al = make_area_light((vec3){-41.404716,-25.552601,17.230299}, (vec3){0.670347,0.612648,-0.418685}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-41.598038,-27.990410,11.597915}, (vec3){0.777166,0.048277,0.627441}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-50.342281,-27.585949,8.561733}, (vec3){0.019471,0.998736,0.046332}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-51.234867,-29.861887,18.782742}, (vec3){0.928283,0.356914,0.104418}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-53.507530,-44.498020,58.075027}, (vec3){0.499710,0.322956,-0.803734}, 0, 3, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-52.430721,-45.030994,57.389839}, (vec3){0.153695,0.987882,-0.021588}, 0, 3, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-54.763268,-44.544991,56.614269}, (vec3){0.361426,-0.880400,0.307031}, 0, 3, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-54.953632,-37.441334,48.620686}, (vec3){0.361426,-0.880400,0.307031}, 0, 3, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-55.503185,-36.335697,41.867279}, (vec3){0.361426,-0.880400,0.307031}, 0, 3, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-51.921284,-36.335697,28.906906}, (vec3){0.361426,-0.880400,0.307031}, 0, 3, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-63.970303,-44.956726,44.704014}, (vec3){0.978384,-0.140130,0.152077}, 0, 3, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-64.882462,-40.227318,41.867706}, (vec3){0.421297,-0.123609,0.898460}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-74.266083,-43.140652,46.926392}, (vec3){0.968323,0.226859,0.104330}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-77.892746,-48.945065,58.133301}, (vec3){0.318402,0.794231,-0.517511}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-68.617020,-49.728664,59.688347}, (vec3){-0.586223,0.325584,-0.741847}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-91.659973,-48.027626,53.720020}, (vec3){0.828976,0.343906,0.441052}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-91.013626,-50.113625,61.160892}, (vec3){0.380150,0.666013,-0.641804}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-115.872803,-50.236839,56.625710}, (vec3){0.925079,0.333452,0.181764}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-124.512474,-47.042686,45.250530}, (vec3){0.186070,-0.076629,0.979544}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-123.844398,-47.042686,46.252995}, (vec3){-0.881693,-0.270513,0.386574}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-121.467407,-45.102448,39.137939}, (vec3){-0.350158,-0.265161,0.898376}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-129.815063,-44.587189,32.666492}, (vec3){0.594137,0.530684,0.604463}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-147.094437,-49.733555,30.416332}, (vec3){0.633680,0.387848,0.669346}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-136.609558,-44.788708,20.543320}, (vec3){-0.257390,-0.054455,0.964772}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-134.357697,-40.817795,14.959087}, (vec3){-0.592420,0.328209,0.735742}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-139.663559,-43.068546,-4.898956}, (vec3){0.279816,0.293892,0.913964}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-123.765114,-40.580875,24.967489}, (vec3){-0.738921,-0.240975,0.629227}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-117.609238,-37.215092,17.910633}, (vec3){-0.724030,-0.129120,0.677576}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-121.502281,-37.044014,12.239734}, (vec3){0.146567,0.312421,0.938569}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-116.024429,-35.898224,-0.170543}, (vec3){-0.496506,0.042727,0.866981}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-119.912888,-36.892391,-9.434753}, (vec3){-0.496506,0.042727,0.866981}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-108.710068,-35.095856,-10.361274}, (vec3){-0.955483,0.014962,0.294668}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-105.453140,-33.860622,-17.034023}, (vec3){-0.584489,0.076000,0.807834}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-89.299095,-31.435440,-20.530695}, (vec3){-0.999905,0.003852,0.013205}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-99.753113,-37.886646,-34.394264}, (vec3){0.171315,0.614841,0.769819}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-109.918716,-48.349567,-52.797836}, (vec3){0.258626,0.565528,0.783128}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-121.800095,-52.427113,-72.535889}, (vec3){-0.873349,0.202443,0.443033}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-87.665199,-53.174778,-84.283195}, (vec3){-0.873349,0.202443,0.443033}, 0, 3, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-73.277298,-50.021652,-80.138359}, (vec3){-0.814451,-0.222060,-0.536059}, 0, 3, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-74.951820,-50.589027,-82.566246}, (vec3){0.000000,-1.000000,-0.000000}, 0, 3, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-77.535347,-50.589027,-82.646431}, (vec3){0.000000,-1.000000,-0.000000}, 0, 3, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-80.208626,-50.589027,-82.729408}, (vec3){0.000000,-1.000000,-0.000000}, 0, 3, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-74.232101,-52.642143,-92.619057}, (vec3){-0.544786,0.438747,0.714639}, 0, 3, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-63.718197,-52.020119,-94.820786}, (vec3){-0.544786,0.438747,0.714639}, 0, 3, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-57.362141,-54.733738,-101.029610}, (vec3){-0.908484,0.023646,-0.417249}, 0, 3, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-41.126953,-52.315620,-88.547020}, (vec3){-0.274505,-0.367530,-0.888577}, 0, 3, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-36.311897,-54.062672,-94.662071}, (vec3){-0.974091,0.202787,-0.100117}, 0, 3, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-40.306095,-53.598690,-94.108971}, (vec3){0.793583,-0.103942,-0.599518}, 0, 5, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-29.506622,-52.541500,-86.343254}, (vec3){-0.182172,0.092958,-0.978863}, 0, 5, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-25.625540,-49.230366,-78.060890}, (vec3){0.420022,-0.390663,-0.819124}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-23.223291,-49.230366,-82.361046}, (vec3){-0.576534,0.789356,-0.211012}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-9.100097,-49.230366,-78.123520}, (vec3){-0.766971,-0.092884,-0.634923}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-14.582089,-46.060249,-70.562126}, (vec3){0.062886,-0.413552,-0.908306}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-24.489586,-37.344414,-61.836494}, (vec3){0.062886,-0.413552,-0.908306}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-33.554474,-36.034813,-64.655312}, (vec3){0.935507,0.090192,-0.341603}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-44.741936,-36.034813,-72.549347}, (vec3){0.302939,0.291572,-0.907311}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-51.457577,-35.772980,-89.486351}, (vec3){0.048478,0.963939,-0.261672}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-51.382626,-48.358646,-94.509697}, (vec3){0.550858,-0.134278,-0.823726}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-42.035149,-70.414902,99.849632}, (vec3){-0.029137,-0.148028,0.988554}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-49.824089,-69.618454,92.720306}, (vec3){-0.029137,-0.148028,0.988554}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-36.277981,-60.869881,81.900368}, (vec3){-0.750044,-0.504782,0.427351}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-15.555871,-44.364609,79.801437}, (vec3){-0.750044,-0.504782,0.427351}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-17.998047,-43.508442,68.779533}, (vec3){0.063891,0.104015,0.992521}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-28.570158,-47.602360,61.886845}, (vec3){0.526379,0.561229,0.638707}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-42.590244,-51.895718,65.812889}, (vec3){0.910282,0.336401,-0.241290}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-37.854904,-44.462784,49.185081}, (vec3){0.172728,-0.315322,0.933133}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-23.791145,-35.443649,44.484119}, (vec3){-0.859327,-0.153520,0.487840}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-23.400446,-30.793144,37.432289}, (vec3){-0.492863,-0.229834,0.839203}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-5.402559,-15.679673,24.080116}, (vec3){-0.889034,0.278260,0.363579}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){16.224276,-12.121444,20.387159}, (vec3){-0.828434,0.259532,0.496327}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){12.629716,-15.518938,22.607395}, (vec3){0.518871,-0.004129,0.854843}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-5.673455,-8.984480,22.003946}, (vec3){0.586338,0.533334,0.609723}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-17.392838,-6.229719,2.450006}, (vec3){0.083971,0.931548,0.353789}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){4.489325,-41.582645,112.645683}, (vec3){-0.085994,0.246094,0.965424}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){11.501933,-46.273983,118.997795}, (vec3){0.219943,0.880268,-0.420421}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){27.651237,-46.866280,115.879921}, (vec3){-0.582277,0.662123,-0.471748}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){27.917965,-45.666214,108.834427}, (vec3){-0.842551,0.040304,0.537107}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){28.118212,-34.878872,96.436569}, (vec3){-0.093662,-0.307403,0.946959}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){20.931847,-29.828487,85.004135}, (vec3){0.276002,0.487967,0.828077}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){18.602524,-22.244993,64.756958}, (vec3){0.276002,0.487967,0.828077}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){18.344563,-16.119905,45.621910}, (vec3){-0.104704,0.842241,0.528836}, 0, 3, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){30.754782,-18.738426,33.311623}, (vec3){-0.104704,0.842241,0.528836}, 0, 3, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){44.837467,-13.928511,23.883783}, (vec3){-0.104704,0.842241,0.528836}, 0, 3, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){56.784031,-24.554703,12.355318}, (vec3){-0.104704,0.842241,0.528836}, 0, 3, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){52.494011,-32.426693,17.369560}, (vec3){-0.690923,0.079121,0.718586}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){69.370918,-32.426693,15.911938}, (vec3){-0.829483,0.248785,0.500064}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){90.823738,-31.857428,8.041292}, (vec3){-0.731694,0.310116,0.607002}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){81.647148,-28.968674,-1.389739}, (vec3){0.691057,-0.134278,0.710218}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){75.632599,-22.406855,-8.822274}, (vec3){0.669124,0.542700,0.507691}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){86.724869,-27.160694,-25.762806}, (vec3){0.669124,0.542700,0.507691}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){61.774349,-27.160694,-34.069202}, (vec3){0.983825,0.098488,-0.149626}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){57.951313,-24.885433,-25.119923}, (vec3){0.965701,0.104015,-0.237914}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){32.521744,-10.577869,-34.122124}, (vec3){0.995990,-0.087351,0.019308}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){31.535631,-9.852077,-22.141356}, (vec3){0.175816,0.668345,-0.722775}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){30.880028,-9.408706,14.336282}, (vec3){0.318680,0.601894,-0.732234}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){42.548466,-23.768517,37.807720}, (vec3){0.256693,0.617307,-0.743667}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){58.014439,-34.923145,42.301361}, (vec3){0.256693,0.617307,-0.743667}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){83.683815,-63.563744,57.287140}, (vec3){-0.248321,0.208225,-0.946033}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){72.104721,-62.213360,108.487885}, (vec3){0.717313,-0.227130,0.658691}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){67.646744,-62.213360,109.861580}, (vec3){0.141100,0.978963,-0.147385}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){66.857613,-58.062374,101.174675}, (vec3){0.902965,-0.380411,0.199856}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){70.080818,-58.062374,97.140526}, (vec3){0.093010,-0.478173,0.873327}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){62.713848,-48.955441,93.563789}, (vec3){0.964351,0.029200,0.263012}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){63.035854,-44.239906,85.374878}, (vec3){0.862959,-0.380411,0.332549}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){43.655537,-27.553997,72.776237}, (vec3){0.563387,-0.475732,0.675481}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){45.211021,-48.874180,108.515221}, (vec3){0.157785,0.806059,-0.570414}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){40.290253,-51.035942,117.824631}, (vec3){0.994590,0.031902,0.098859}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){14.771205,-42.061867,108.687393}, (vec3){0.440786,-0.131598,0.887913}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){33.586220,-26.384285,86.419037}, (vec3){0.060943,-0.208224,0.976181}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){40.337349,-21.205442,74.782631}, (vec3){0.499881,0.388105,0.774270}, 0, 3, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-32.767750,-24.896996,28.187532}, (vec3){-0.670935,-0.202787,0.713249}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-25.157635,-17.412888,15.889715}, (vec3){-0.474582,-0.219079,0.852512}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            al = make_area_light((vec3){-26.575705,-3.489005,-23.941166}, (vec3){-0.196450,0.426158,0.883061}, 0, 4, -1.0f, -1.0f, -1.0f, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);

            // Copy to mirrored suntemples (0,0,0) -> (250,0,0), (0, 0, 250), and (250, 0, 250)
            int initial_num_arear_lights = (int)array_length(&program.area_lights, sizeof(AreaLight));
            int initial_num_point_lights = (int)array_length(&program.point_lights, sizeof(PointLight));
            for (int mirror_z = 0; mirror_z < 4; ++mirror_z)
            {
                for (int mirror_x = 0; mirror_x < 4; ++mirror_x)
                {
                    if (mirror_z == 0 && mirror_x == 0)
                    {
                        continue;
                    }

                    for (int al_i = 0; al_i < initial_num_arear_lights; ++al_i)
                    {
                        AreaLight* ith_al = get_element(&program.area_lights, sizeof(AreaLight), al_i);
                        memcpy(&al, ith_al, sizeof(AreaLight));

                        mat4 transform = GLM_MAT4_IDENTITY_INIT;
                        vec3 translation_vec = { 250.0f * mirror_x, 0.0f, -250.0f * mirror_z };
                        glm_translate(transform, translation_vec);
                        transform_area_light(&al, transform);

                        push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
                    }

                    // Point lights too for fun
                    PointLight pl;
                    for (int pl_i = 0; pl_i < initial_num_point_lights; ++pl_i)
                    {
                        PointLight* ith_pl = get_element(&program.point_lights, sizeof(PointLight), pl_i);
                        memcpy(&pl, ith_pl, sizeof(PointLight));

                        vec3 translation_vec = { 250.0f * mirror_x, 0.0f, -250.0f * mirror_z };
                        glm_vec3_add(pl.position, translation_vec, pl.position);                        
                        
                        push_element_copy(&program.point_lights, sizeof(PointLight), &pl);
                    }
                }
            }
        }
    }
    if (scene_id == 3)
    {
        AreaLight al;
        // al = make_area_light((vec3){-5.528228,0.563686,-4.714774}, (vec3){-0.639877,-0.004516,-0.768464}, 0, 5, -1.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
        // al = make_area_light((vec3){2.444456,2.973819,-5.098126}, (vec3){0.888351,-0.110246,0.445734}, 0, 4, -1.0f);push_element_copy(   &program.area_lights, sizeof(AreaLight), &al);
        // al = make_area_light((vec3){2.437656,0.434195,1.099524}, (vec3){-0.686625,-0.628744,-0.365003}, 0, 3, -1.0f);push_element_copy(  &program.area_lights, sizeof(AreaLight), &al);
        // al = make_area_light((vec3){-6.828334,2.708745,-5.153105}, (vec3){-0.997019,0.013983,-0.075884}, 0, , -1.0f3);push_element_copy( &program.area_lights, sizeof(AreaLight), &al);
        // al = make_area_light((vec3){-4.408458,5.684846,-6.819602}, (vec3){-0.987434,-0.090130,0.129809}, 0, 4, -1.0f);push_element_copy( &program.area_lights, sizeof(AreaLight), &al);
        // al = make_area_light((vec3){ 0.0f, 0.0f, 0.0f }, (vec3){ 0.0f, 1.0f, 0.0f }, 1, 4); push_element_copy(&program.area_lights, sizeof(AreaLight), &al);

        al = make_area_light((vec3){4.310672,15.536465,-71.449356}, (vec3){0.264739,0.085128,0.960555}, 0, 3, 0.3f   , 20.0f, 3.0f, 3.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
        al = make_area_light((vec3){12.480458,9.082880,-69.066299}, (vec3){0.264739,0.085128,0.960555}, 0, 4, 0.2f    , 20.0f, 7.0f, 2.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
        al = make_area_light((vec3){-3.481347,10.381870,-110.580444}, (vec3){-0.246875,-0.681156,0.689259}, 0, 3, 0.4f, 10.0f, 1.0f, 4.0f);push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
    }

    size_t len_pl = array_length(&program.point_lights, sizeof(PointLight));
    size_t len_al = array_length(&program.area_lights, sizeof(AreaLight));
    printf("Point Light Array len: %d\n", (int)len_pl);
    printf("Area Light Array len: %d\n", (int)len_al);

    // Init directional lighting
    if (scene_id >= 0 && scene_id <= 2)
    {
        // Turn off directional lights for sponza, suntemple and lostempire
        out_loaded_scene->sun_color[0] = 1.0f;
        out_loaded_scene->sun_color[1] = 1.0f;
        out_loaded_scene->sun_color[2] = 1.0f;
        // out_loaded_scene->sun_intensity = 0.0f;  // Turn off sun
        out_loaded_scene->sun_intensity = 0.1f;
        out_loaded_scene->sun_direction[0] = 1.0f;
        out_loaded_scene->sun_direction[1] = 1.0f;
        out_loaded_scene->sun_direction[2] = 1.0f;
    }
    else
    {
        // Turn on directional light for other scenes where we don't have any initial point lights
        // Give sun light to other scenes since no default point lights to light the scene
        out_loaded_scene->sun_direction[0] = 0.5f;
        out_loaded_scene->sun_direction[1] = 1.0f;
        out_loaded_scene->sun_direction[2] = 0.3f;
        out_loaded_scene->sun_intensity = 0.5f;
        out_loaded_scene->sun_color[0] = 1.0f;
        out_loaded_scene->sun_color[1] = 1.0f;
        out_loaded_scene->sun_color[2] = 1.0f;
    }

    // printf("TEMP: Deleting point lights for now\n");
    // free_array(&program.point_lights);
    // program.point_lights = create_array(1 * sizeof(PointLight));

    init_global_renderer_buffers();
}

u32
compile_shader_type(GLenum gl_shader_type, const char* shader_src, const char* opengl_debug_name)
{
    u32 shader;

    shader = glCreateShader(gl_shader_type);
    glObjectLabel(GL_SHADER, shader, -1, opengl_debug_name);
    glShaderSource(shader, 1, (const char* const*)&shader_src, NULL);
    glCompileShader(shader);

    // Check for vertex shader compile errors
    char info_log[1024];
    int compilation_success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compilation_success);
    
    if (!compilation_success)
    {
        glGetShaderInfoLog(shader, 1024, NULL, info_log);
        printf("%s", info_log);
    }

    return shader;
}

u32
compile_opaque_program(const char* vertex_src, const char* fragment_src, const char* opengl_debug_name)
{
    assert(vertex_src && fragment_src);

    char* vertex_shader_debug_name = malloc_strcat(opengl_debug_name, "-vertex");
    char* fragment_shader_debug_name = malloc_strcat(opengl_debug_name, "-fragment");

    u32 vertex_shader = compile_shader_type(GL_VERTEX_SHADER, vertex_src, vertex_shader_debug_name);
    u32 fragment_shader = compile_shader_type(GL_FRAGMENT_SHADER, fragment_src, fragment_shader_debug_name);

    free(vertex_shader_debug_name);
    free(fragment_shader_debug_name);

    u32 opaque_program = glCreateProgram();
    glObjectLabel(GL_PROGRAM, opaque_program, -1, opengl_debug_name);

    glAttachShader(opaque_program, vertex_shader);
    glAttachShader(opaque_program, fragment_shader);
    glLinkProgram(opaque_program);

    // Check for link errors
    char info_log[1024];
    int program_success;
    glGetProgramiv(opaque_program, GL_LINK_STATUS, &program_success);
    
    if (!program_success)
    {
        glGetProgramInfoLog(opaque_program, 1024, NULL, info_log);
        printf("%s", info_log);
    }

    // Validate the program
    glValidateProgram(opaque_program);
    glGetProgramiv(opaque_program, GL_VALIDATE_STATUS, &program_success);
    if (!program_success) {
        glGetProgramInfoLog(opaque_program, 1024, NULL, info_log);
        printf("Shader Program Validation Error: %s\n", info_log);
    }

    glDetachShader(opaque_program, vertex_shader);
    glDetachShader(opaque_program, fragment_shader);

    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);

    return opaque_program;
}

u32
compile_compute_program(const char* compute_src, const char* opengl_debug_name)
{
    assert(compute_src);

    char* compute_shader_debug_name = malloc_strcat(opengl_debug_name, "-compute");

    u32 compute_shader = compile_shader_type(GL_COMPUTE_SHADER, compute_src, compute_shader_debug_name);
    free(compute_shader_debug_name);

    u32 opaque_program = glCreateProgram();
    glObjectLabel(GL_PROGRAM, opaque_program, -1, opengl_debug_name);

    glAttachShader(opaque_program, compute_shader);
    glLinkProgram(opaque_program);

    // Check for link errors
    char info_log[1024];
    int program_success;
    glGetProgramiv(opaque_program, GL_LINK_STATUS, &program_success);
    
    if (!program_success)
    {
        glGetProgramInfoLog(opaque_program, 1024, NULL, info_log);
        printf("%s", info_log);
    }

    glDetachShader(opaque_program, compute_shader);
    glDeleteShader(compute_shader);

    return opaque_program;
}

u32
load_shader_from_files(const char* vertex_filename, const char* fragment_filename, const char* opengl_debug_name)
{
    char* vertex_src = load_text_file(vertex_filename);
    if (vertex_src == NULL)
    {
        printf("Couldn't open shader file: %s\n", vertex_filename);
        exit(1);
    }

    char* fragment_src  = load_text_file(fragment_filename);
    if (fragment_src == NULL)
    {
        printf("Couldn't open shader file: %s\n", fragment_filename);
        exit(1);
    }

    u32 opaque_program = compile_opaque_program(vertex_src, fragment_src, opengl_debug_name);
    free(vertex_src);
    free(fragment_src);

    return opaque_program;
}

// I needed a function to dynamically set compile time flags for shaders
// specifically #define ENABLE_CLUSTERED_SHADING, however the prepended header
// must go after the "#version 460 core" directive, so we put it on the second line
// of the shader source file, instead of prepending the header completely
char*
insert_glsl_header_after_version_directive(const char* src, const char* header)
{
    const char* newline = strchr(src, '\n');
    if (newline == NULL)
    {
        printf("Shader source invalid, requires newline after #version directive.\n");
        exit(1);
    }

    size_t header_len = strlen(header);
    size_t version_directive_line_len = newline - src + 1;  // Include +1 room for newline char
    size_t remaining_len = strlen(src) - version_directive_line_len;

    size_t final_size = version_directive_line_len + header_len + 1 + remaining_len + 1;  // +1 for header's \n, +1 for final \0
    char* inserted_src = malloc(final_size);

    // Copy the #version directive line
    memcpy(inserted_src, src, version_directive_line_len);

    // Copy in header and append a newline to it
    memcpy(inserted_src + version_directive_line_len, header, header_len);
    inserted_src[version_directive_line_len + header_len] = '\n';

    // Append the rest of the shader source and null terminate
    memcpy(inserted_src + version_directive_line_len + header_len + 1, src + version_directive_line_len, remaining_len);
    inserted_src[final_size - 1] = '\0';

    return inserted_src;
}

u32
load_shader_from_files_with_header(const char* vertex_filename, const char* fragment_filename, const char* opengl_debug_name, const char* header_text)
{
    char* vertex_src = load_text_file(vertex_filename);
    if (vertex_src == NULL)
    {
        printf("Couldn't open shader file: %s\n", vertex_filename);
        exit(1);
    }

    char* fragment_src  = load_text_file(fragment_filename);
    if (fragment_src == NULL)
    {
        printf("Couldn't open shader file: %s\n", fragment_filename);
        exit(1);
    }

    char* vertex_src_with_header = insert_glsl_header_after_version_directive(vertex_src, header_text);
    char* fragment_src_with_header = insert_glsl_header_after_version_directive(fragment_src, header_text);
    
    u32 opaque_program = compile_opaque_program(vertex_src_with_header, fragment_src_with_header, opengl_debug_name);
    free(vertex_src);
    free(fragment_src);
    free(vertex_src_with_header);
    free(fragment_src_with_header);

    return opaque_program;
}

u32
load_compute_shader_from_file_with_header(const char* compute_filename, const char* opengl_debug_name, const char* header_text)
{
    char* compute_src = load_text_file(compute_filename);
    if (compute_src == NULL)
    {
        printf("Couldn't open shader file: %s\n", compute_filename);
        exit(1);
    }
    
    char* compute_src_with_header = insert_glsl_header_after_version_directive(compute_src, header_text);
    
    u32 opaque_program = compile_compute_program(compute_src_with_header, opengl_debug_name);
    free(compute_src);
    free(compute_src_with_header);

    return opaque_program;
}

//////////////////

void
update_free_camera(FreeCamera* cam)
{
    cam->width = program.w;
    cam->height = program.h;

    if (program.keydown_zoom_in)
    {
        cam->fov_y = glm_rad(15.0f);
    }
    else
    {
        cam->fov_y = glm_rad(60.0f);
    }

    if (program.mouse_capture_on)
    {
        // Look around with mouse:
        const f32 sensitivity = 2.0f;
        {
            f32 mouse_motion_x = sensitivity * ((f32)program.mouse_relative_x / (f32)program.w);
            f32 mouse_motion_y = sensitivity * ((f32)program.mouse_relative_y / (f32)program.h);

            cam->yaw += mouse_motion_x;
            cam->pitch += mouse_motion_y;

            // Clamp pitch to avoid going upside-down
            if (cam->pitch < -PI/2.0f)     cam->pitch = -PI/2.0f;
            else if (cam->pitch > PI/2.0f) cam->pitch =  PI/2.0f;

            // Wrap yaw to [0, 2PI)
            if (cam->yaw < 0.0f) cam->yaw += 2.0f * PI;
            cam->yaw = fmodf(cam->yaw, 2.0f * PI);
        }

        // Move camera
        {
            // Buttons:
            int f = program.keydown_forward;
            int b = program.keydown_backward;
            int l = program.keydown_left;
            int r = program.keydown_right;
            int up = program.keydown_up;
            int down = program.keydown_down;

            f32 speed = 5.0f;
            if (program.keydown_sprint)
            {
                speed *= 15.0f;
            }

            f32 pos_increment = speed * program.dt;
            if ((f != b) && (l != r))
            {
                // Normalize speed if going diagonally
                pos_increment *= (f32)(1.0 / sqrt(2.0));
            }
            
            f32 dx = pos_increment * sinf(cam->yaw);
            f32 dz = pos_increment * cosf(cam->yaw);
            cam->pos[0] += dx * (f-b) + dz * (r-l);
            cam->pos[2] += dx * (r-l) + dz * (b-f);

            cam->pos[1] += speed * program.dt * (f32)(up - down);
        }
    }

    // View matrix: Move world to camera position and orientation
    mat4 view = GLM_MAT4_IDENTITY_INIT;
    glm_rotate(view, cam->pitch, (vec3){ 1.0f, 0.0f, 0.0f });
    glm_rotate(view, cam->yaw, (vec3){ 0.0f, 1.0f, 0.0f });
    glm_translate(view, (vec3){ -cam->pos[0], -cam->pos[1], -cam->pos[2] });

    // Projection matrix: Project world space to screen space
    mat4 proj = GLM_MAT4_IDENTITY_INIT;
    glm_perspective(cam->fov_y, (float)cam->width / (float)cam->height, cam->near_plane, cam->far_plane, proj);

    // Camera matrix: proj*view
    glm_mat4_mul(proj, view, cam->camera_matrix);
    glm_mat4_copy(view, cam->view_matrix);
    glm_mat4_copy(proj, cam->projection_matrix);
}

void APIENTRY
opengl_callback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* user_param)
{
    (void)source;
    (void)type;
    (void)id;
    (void)length;
    (void)user_param;

    // Display OpenGL warning
    if (severity == GL_DEBUG_SEVERITY_HIGH || severity == GL_DEBUG_SEVERITY_MEDIUM || severity == GL_DEBUG_SEVERITY_LOW)
    {
        if (severity == GL_DEBUG_SEVERITY_HIGH)         printf("OpenGL [HIGH]: ");
        else if (severity == GL_DEBUG_SEVERITY_MEDIUM)  printf("OpenGL [MEDIUM]: ");
        else if (severity == GL_DEBUG_SEVERITY_LOW)     printf("OpenGL [LOW]: ");
        printf("%s\n", message);
    }

    if (severity == GL_DEBUG_SEVERITY_HIGH)
    {
        printf("OpenGL error severity was high and needs fixing!\n");

        // Crash program so we can backtrace
        *((volatile int*)0) = 0;
        // abort();
    }
}

void
reload_shaders(b32 only_reload_pbr_shaders)
{
    printf("Compiling with cluster_max_lights=%d...\n", program.max_lights_per_cluster);

    //
    // Silly hacky metaprogramming to add the following to the shaders:
    // add #define ENABLE_CLUSTERED_SHADING to top of headers when enabled
    // add #define SHOW_NORMALS to top of pbr shader when enabled
    // add #define CLUSTER_MAX_LIGHTS int(program.max_lights_per_cluster) to header of all shaders
    //
    const char* header_a = "#define ENABLE_CLUSTERED_SHADING";
    const char* header_b = "";
    const char* header_normals_a = "#define ENABLE_CLUSTERED_SHADING\n#define SHOW_NORMALS";
    const char* header_normals_b = "#define SHOW_NORMALS";
    const char* base_header_text;
    if (program.is_clustered_shading_enabled)
    {
        if (program.render_just_normals) base_header_text = header_normals_a;
        else base_header_text = header_a;
    }
    else
    {
        if (program.render_just_normals) base_header_text = header_normals_b;
        else base_header_text = header_b;
    }

    char light_ops_allow_char = ' ';
    if (!program.is_light_op_counting_enabled)
    {
        light_ops_allow_char = '/';  // Stupid way to comment out #define COUNT_LIGHT_OPS
    }

    #ifdef INTEGRATED_GPU
    char integrated_gpu_char = ' ';
    #else
    char integrated_gpu_char = '/';
    #endif

    char header_text[1024] = { 0 };  // For all shaders
    snprintf(header_text, sizeof(header_text),
            "%s\n#define CLUSTER_GRID_SIZE_X " xstr(CLUSTER_GRID_SIZE_X)
            "\n#define CLUSTER_GRID_SIZE_Y " xstr(CLUSTER_GRID_SIZE_Y)
            "\n#define CLUSTER_GRID_SIZE_Z " xstr(CLUSTER_GRID_SIZE_Z)
            "\n#define CLUSTER_NORMALS_COUNT " xstr(CLUSTER_NORMALS_COUNT)
            "\n#define CLUSTER_MAX_LIGHTS %d"
            "\n%c%c#define COUNT_LIGHT_OPS"  // Stupid way to comment out this line according to a boolean
            "\n#define MAX_UNCLIPPED_NGON %d"
            "\n%c%c#define INTEGRATED_GPU",
        base_header_text, program.max_lights_per_cluster, light_ops_allow_char, light_ops_allow_char, MAX_UNCLIPPED_NGON, integrated_gpu_char, integrated_gpu_char);


    // Compile
    if (program.shader_pbr_opaque) glDeleteProgram(program.shader_pbr_opaque);
    program.shader_pbr_opaque = load_shader_from_files_with_header("shader_src/pbr.vert", "shader_src/pbr.frag", "pbr_shader_opaque_pass", header_text);
    // strncat(header_text, "#define TRANSPARENT_PASS\n", sizeof(header_text) - strlen(header_text) - 1);
    // program.shader_pbr_transparent = load_shader_from_files_with_header("shader_src/pbr.vert", "shader_src/pbr.frag", "pbr_shader_transparent_pass", header_text);

    if (!only_reload_pbr_shaders)
    {
        if (program.shader_area_light_polygons) glDeleteProgram(program.shader_area_light_polygons);
        if (program.shader_compute_clusters) glDeleteProgram(program.shader_compute_clusters);
        if (program.shader_light_assignment) glDeleteProgram(program.shader_light_assignment);

        program.shader_area_light_polygons = load_shader_from_files("shader_src/polygon.vert", "shader_src/polygon.frag", "polygon_shader");
        program.shader_compute_clusters = load_compute_shader_from_file_with_header("shader_src/voxel_clusters_viewspace.comp", "compute_clusters_shader", header_text);
        program.shader_light_assignment = load_compute_shader_from_file_with_header("shader_src/lights_to_clusters.comp", "light_assignment_shader", header_text);
    }

    printf("  ...Complete.\n");
}

void
window_size_callback(GLFWwindow* window, int width, int height)
{
    // Handle window minimized
    if (width == 0 || height == 0)
    {
        glfwSetWindowShouldClose(window, GLFW_FALSE);
        program.is_minimized = 1;
        return;
    }

    program.is_minimized = 0;

    program.w = width;
    program.h = height;
    program.aspect_ratio = (f32)program.w / (f32)program.h;


    glViewport(0, 0, width, height);
}

void
key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    /* Key Bindings:
     *      Escape - Quit
     *      Enter  - Print Player's Position
     *      E      - Toggle mouse capture
     *      WASD   - Movement
     *      Space  - Up
     *      Shift  - Down
     *      Ctrl   - Sprint
     *      C      - Zoom In
     *      F1     - Toggle wireframe mode
     *      F2     - Toggle cluster visualisation
     *      B      - Toggle GUI for nicer screenshots
     *      
     *      Enter     - Spawn area light
     *      Alt+Enter - Spawn point light
     */

    (void)scancode;

    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(window, 1);
    }

    if (key == GLFW_KEY_ENTER && action == GLFW_PRESS)
    {
        printf("{ %f, %f, %f }, pitch=%f, yaw=%f,\n", program.cam.pos[0], program.cam.pos[1], program.cam.pos[2], program.cam.pitch, program.cam.yaw);

        int alt_key = mods & GLFW_MOD_ALT;

        if (alt_key)
        {
            // Spawn point light at position
            PointLight pl = { 0 };
            glm_vec3_copy(program.cam.pos, pl.position);
            pl.color[0] = rng_rangef(0.1f, 1.0f);
            pl.color[1] = rng_rangef(0.1f, 1.0f);
            pl.color[2] = rng_rangef(0.1f, 1.0f);
            pl.intensity = 10.0f / glm_vec3_norm(pl.color);

            push_element_copy(&program.point_lights, sizeof(PointLight), &pl);
        }
        else
        {
            // Spawn area light at position        
            int n = (int)program.last_number_key;
            if (n == 6)
                n = 10;  // 6 key is used for star
            
            vec3 true_forward = {
                program.cam.view_matrix[0][2],
                program.cam.view_matrix[1][2],
                program.cam.view_matrix[2][2],
            };
            
            int double_sided = 0;
            AreaLight al = make_area_light(program.cam.pos, true_forward, double_sided, n, -1.0f, -1.0f, -1.0f, -1.0f);
            push_element_copy(&program.area_lights, sizeof(AreaLight), &al);
            
            // Output code snippet to regenerate the lights
            printf("AreaLight al = make_area_light((vec3){%f,%f,%f}, (vec3){%f,%f,%f}, %d, %d, -1.0f, -1.0f, -1.0f, -1.0f);",
                program.cam.pos[0],program.cam.pos[1],program.cam.pos[2],
                true_forward[0],true_forward[1],true_forward[2], double_sided, n);
            printf("push_element_copy(&program.area_lights, sizeof(AreaLight), &al);\n");
        }
    }

    if (action == GLFW_PRESS)
    {
        if (key == GLFW_KEY_3)
            program.last_number_key = 3;
        else if (key == GLFW_KEY_4)
            program.last_number_key = 4;
        else if (key == GLFW_KEY_5)
            program.last_number_key = 5;
        else if (key == GLFW_KEY_6)
            program.last_number_key = 6;
        
        // Teleport to same spot one keys:
        if (key == GLFW_KEY_7)
        {
            glm_vec3_copy((vec3){ -0.286521, -3.771567, -88.778755 }, program.cam.pos);
            program.cam.pitch = -0.067008f;
            program.cam.yaw = 3.043161f;
        }
        else if (key == GLFW_KEY_8)
        {
            glm_vec3_copy((vec3){ 320.440155, 219.672180, 197.722351 }, program.cam.pos);
            program.cam.pitch = 0.726073;
            program.cam.yaw = 6.271652;
        }
    }

    if (key == GLFW_KEY_E && action == GLFW_PRESS)
    {
        program.mouse_capture_on = !program.mouse_capture_on;

        // Toggle mouse mode
        glfwSetInputMode(window, GLFW_CURSOR, program.mouse_capture_on ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
    }

    if (key == GLFW_KEY_F1 && action == GLFW_PRESS)
    {
        program.render_as_wireframe = !program.render_as_wireframe;
        if (program.render_as_wireframe)
        {
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        }
        else
        {
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        }
    }

    if (key == GLFW_KEY_F2 && action == GLFW_PRESS)
    {
        program.render_just_normals = !program.render_just_normals;
        reload_shaders(0);
    }

    if (key == GLFW_KEY_F3 && action == GLFW_PRESS)
    {
        program.is_clustered_shading_enabled = !program.is_clustered_shading_enabled;
        reload_shaders(0);
    }

    if (key == GLFW_KEY_F4 && action == GLFW_PRESS)
    {
        program.is_light_op_counting_enabled = !program.is_light_op_counting_enabled;
        reload_shaders(0);
    }

    if (action == GLFW_PRESS)
    {
        switch (key)
        {
            case GLFW_KEY_W:
                program.keydown_forward = 1;
                break;
            case GLFW_KEY_A:
                program.keydown_left = 1;
                break;
            case GLFW_KEY_S:
                program.keydown_backward = 1;
                break;
            case GLFW_KEY_D:
                program.keydown_right = 1;
                break;
            
            case GLFW_KEY_SPACE:
                program.keydown_up = 1;
                break;
            case GLFW_KEY_LEFT_SHIFT:
                program.keydown_down = 1;
                break;

            case GLFW_KEY_LEFT_CONTROL:
                program.keydown_sprint = 1;
                break;
            case GLFW_KEY_C:
                program.keydown_zoom_in = 1;
                break;
            
            case GLFW_KEY_B:
                program.keytoggle_disable_gui = !program.keytoggle_disable_gui;
                break;
        }
    }
    else if (action == GLFW_RELEASE)
    {
        switch (key)
        {
            case GLFW_KEY_W:
                program.keydown_forward = 0;
                break;
            case GLFW_KEY_A:
                program.keydown_left = 0;
                break;
            case GLFW_KEY_S:
                program.keydown_backward = 0;
                break;
            case GLFW_KEY_D:
                program.keydown_right = 0;
                break;
            
            case GLFW_KEY_SPACE:
                program.keydown_up = 0;
                break;
            case GLFW_KEY_LEFT_SHIFT:
                program.keydown_down = 0;
                break;
            
            case GLFW_KEY_LEFT_CONTROL:
                program.keydown_sprint = 0;
                break;
            case GLFW_KEY_C:
                program.keydown_zoom_in = 0;
                break;
        }
    }
}

void
reset_opengl_render_state()
{
    if (program.is_msaa_enabled)
    {
        glEnable(GL_MULTISAMPLE);
    }

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);  // Ordered transparency
    // glBlendFunc(GL_ONE, GL_ONE);
    glDepthMask(GL_TRUE);
    glReadBuffer(GL_BACK);  // Make sure this is set for glCopyTextureSubImage2D after the opaque render pass

    glFrontFace(GL_CCW);  // Counter-clockwise is the default, but I've forgetten this before so I prefer being explicit
    // glClearColor(0.6f, 0.8f, 0.9f, 0.0f);  // Cornflower blue
    glClearColor(0.3f, 0.4f, 0.5f, 0.0f);  // Dark blue
}

int
main(int argc, char** argv)
{
    (void)argc;
    (void)argv;

    const char window_title[] = "COMP3931";
    program.w = 1280;
    program.h = 720;
    program.aspect_ratio = (f32)program.w / (f32)program.h;
    program.frame_counter = 0;
    // program.is_hdr_enabled = 0;  // <- Removed because my laptop only supports r10g10b10a2 instead of rgba16 it so can't work on it rn.
    program.is_msaa_enabled = 0;
    program.is_minimized = 0;
    program.is_clustered_shading_enabled = 1;
    program.max_lights_per_cluster = CLUSTER_DEFAULT_MAX_LIGHTS;

    program.render_as_wireframe = 0;
    program.render_just_normals = 0;

    program.last_number_key = 4;

    program.compute_time_last_frame = 0.0f;

    // Init GLFW, load OpenGL 4.6, and setup nuklear GUI library
    {
        if (!glfwInit())
            exit(-1);
        
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        
        glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_TRUE);
        glfwWindowHint(GLFW_DEPTH_BITS, 24);

        // if (program.is_hdr_enabled)
        // {
        //     glfwWindowHint(GLFW_RED_BITS, 16);
        //     glfwWindowHint(GLFW_GREEN_BITS, 16);
        //     glfwWindowHint(GLFW_BLUE_BITS, 16);
        //     glfwWindowHint(GLFW_ALPHA_BITS, 16);
        // }

        if (program.is_msaa_enabled)
        {
            glfwWindowHint(GLFW_SAMPLES, 4);  // For MSAA with GL_MULTISAMPLE
        }

        program.window = glfwCreateWindow(program.w, program.h, window_title, NULL, NULL);
        if (!program.window)
        {
            glfwTerminate();
            exit(-1);
        }

        glfwMakeContextCurrent(program.window);
        gladLoadGL();
        glfwSwapInterval(0);  // VSYNC OFF

        // int red_bits, green_bits, blue_bits, alpha_bits;
        // glBindFramebuffer(GL_FRAMEBUFFER, 0);
        // glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, GL_BACK_LEFT, GL_FRAMEBUFFER_ATTACHMENT_RED_SIZE, &red_bits);
        // glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, GL_BACK_LEFT, GL_FRAMEBUFFER_ATTACHMENT_GREEN_SIZE, &green_bits);
        // glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, GL_BACK_LEFT, GL_FRAMEBUFFER_ATTACHMENT_BLUE_SIZE, &blue_bits);
        // glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, GL_BACK_LEFT, GL_FRAMEBUFFER_ATTACHMENT_ALPHA_SIZE, &alpha_bits);
        // printf("Framebuffer bits: R=%d, G=%d, B=%d, A=%d\n", red_bits, green_bits, blue_bits, alpha_bits);
        // if (program.is_hdr_enabled && 
        //     (red_bits < 16 || green_bits < 16 || blue_bits < 16 || alpha_bits < 16))
        // {
        //     printf("High Dynamic Range (HDR) window buffers request failed.\n");
        //     program.is_hdr_enabled = 0;
        // }


        program.gui_context = nk_glfw3_init(&program.gui_glfw, program.window, NK_GLFW3_INSTALL_CALLBACKS);
        nk_glfw3_font_stash_begin(&program.gui_glfw, &program.gui_font_atlas);
        nk_glfw3_font_stash_end(&program.gui_glfw);
        set_style(program.gui_context, THEME_DARK);
    }

    // Set GLFW callbacks and settings
    {
        glfwSetWindowSizeCallback(program.window, window_size_callback);
        glfwSetKeyCallback(program.window, key_callback);

        // // Disable mouse acceleration
        // if (glfwRawMouseMotionSupported())
        //     glfwSetInputMode(program.window, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
    }

    // OpenGL renderer setup
    {
        printf("OpenGL\n{\n");
        printf("  Vendor:   %s\n", glGetString(GL_VENDOR));
        printf("  Renderer: %s\n", glGetString(GL_RENDERER));
        printf("  Version:  %s\n}\n", glGetString(GL_VERSION));
        printf("\nWindow Dimensions: (%d, %d)  Aspect ratio: %f\n\n", program.w, program.h, program.aspect_ratio);
        program.driver_name = (const char*)glGetString(GL_RENDERER);

        glEnable(GL_DEBUG_OUTPUT);
        glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
        glDebugMessageCallback(opengl_callback, NULL);
        glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, NULL, 1);

        reset_opengl_render_state();  // This is also called every frame since nuklear affects global GL state
    }

    init_global_renderer_buffers();
    glGenQueries(1, &program.compute_time_query);  // Make time query object

    // Compiler shaders
    {
        program.shader_pbr_opaque = 0;
        program.shader_compute_clusters = 0;
        program.shader_light_assignment = 0;
        reload_shaders(0);
    }

    // Load the LTC distribution matrix table textures
    {
        program.LTC1_texture = load_ltc_matrix_texture(LTC1);
        program.LTC2_texture = load_ltc_matrix_texture(LTC2);
    }

    // Load scene
    {
        // Init camera
        program.cam.near_plane = 0.1f;
        program.cam.far_plane = 1000.0f;
        program.cam.pos[0] = 0.0f;
        program.cam.pos[1] = 1.0f;
        program.cam.pos[2] = 0.0f;
        program.cam.pitch = 0.0f;
        program.cam.yaw = PI/2.0f;

        // Load test scene with point lights
        // load_test_scene(1, &program.scene);
        load_test_scene(3, &program.scene);
        // glm_vec3_copy((vec3){ 0.577642f, -0.921296f, -28.718777f }, program.cam.pos);
        // program.cam.pitch = 0.0f;
        // program.cam.yaw = PI;

        // Add temporary area light
        
    }

    // Initialise frame-based stuff before main-loop
    {
        program.time = glfwGetTime();
        program.dt = 0.0;
        program.mouse_relative_x = 0.0;
        program.mouse_relative_y = 0.0;
    }

    while (!glfwWindowShouldClose(program.window))
    {
        // Update time
        {
            f64 new_time = glfwGetTime();
            program.dt = (f32)(new_time - program.time);
            program.time = new_time;
        }

        glfwPollEvents();
        if (program.is_minimized)
        {
            glfwWaitEvents();
            continue;
        }

        // Calculate average fps per half second
        static double displayed_fps = 0.0f;
        static double displayed_compute_time = 0.0f;
        {
            static int num_frames = 0;
            static double num_seconds = 0.0f;
            num_frames++;
            num_seconds += (double)program.dt;

            static double compute_total = 0.0f;
            compute_total += program.compute_time_last_frame / 1e6;

            if (num_seconds > 0.5f)
            {
                displayed_fps = (double)num_frames / num_seconds;
                displayed_compute_time = compute_total / (double)num_frames;
                displayed_compute_time = program.compute_time_last_frame / 1e6;
                num_frames = 0;
                num_seconds = 0.0f;
                compute_total = 0.0f;
            }
        }

        // Display driver and framerate in window title
        {
            char title[512] = { 0 };
            sprintf(title, "%s Hardware: %s FPS: %f (NO VSYNC) Light Ops: %d", window_title, program.driver_name, displayed_fps, program.last_light_ops_value);
            glfwSetWindowTitle(program.window, title);
        }

        // Update mouse motion
        {
            static double prev_xpos = 0.0;
            static double prev_ypos = 0.0;
            static b32 first_frame = 1;

            double xpos, ypos;
            glfwGetCursorPos(program.window, &xpos, &ypos);

            if (first_frame)
            {
                // No mouse motion on first frame
                prev_xpos = xpos;
                prev_ypos = ypos;
                first_frame = 0;
            }

            program.mouse_relative_x = xpos - prev_xpos;
            program.mouse_relative_y = ypos - prev_ypos;

            prev_xpos = xpos;
            prev_ypos = ypos;
        }
        
        update_free_camera(&program.cam);
        
        // Animate area light intensity
        // for (int i = 0; i < array_length(&program.area_lights, sizeof(AreaLight)); ++i)
        // {
        //     AreaLight* al = get_element(&program.area_lights, sizeof(AreaLight), i);
            
        //     float max_intensity = 50.0f;
        //     static float intensity_velocity = 10.0f;
            
        //     al->color_rgb_intensity_a[3] += intensity_velocity * program.dt;
        //     if (al->color_rgb_intensity_a[3] >= max_intensity || al->color_rgb_intensity_a[3] <= 0.0f)
        //     {
        //         intensity_velocity = -intensity_velocity;
        //     }
        // }

#ifndef DISABLE_GUI
        // Create GUI
        if (!program.keytoggle_disable_gui)
        {
            nk_glfw3_new_frame(&program.gui_glfw);
            
            int nk_flags = 0;  // NK_WINDOW_BORDER|NK_WINDOW_TITLE|NK_WINDOW_MINIMIZABLE|NK_WINDOW_MOVABLE|NK_WINDOW_SCALABLE
            
            // Display compute time query in top left:
            if (nk_begin(program.gui_context, "Performance Stats", nk_rect(10, 10, 200, 75), NK_WINDOW_NO_SCROLLBAR))
            {
                char fps_str[64];
                snprintf(fps_str, sizeof(fps_str), "%.2f fps", displayed_fps);
                nk_layout_row_dynamic(program.gui_context, 20, 1);
                nk_label(program.gui_context, fps_str, NK_TEXT_LEFT);

                char time_str[64];
                snprintf(time_str, sizeof(time_str), "Compute Time: %.2f ms", displayed_compute_time);
                nk_layout_row_dynamic(program.gui_context, 20, 1);
                nk_label(program.gui_context, time_str, NK_TEXT_LEFT);

                char grid_str[64];
                snprintf(grid_str, sizeof(grid_str), "Cluster grid (%d,%d,%d, %d)", CLUSTER_GRID_SIZE_X, CLUSTER_GRID_SIZE_Y, CLUSTER_GRID_SIZE_Z, CLUSTER_NORMALS_COUNT);
                nk_layout_row_dynamic(program.gui_context, 20, 1);
                nk_label(program.gui_context, grid_str, NK_TEXT_LEFT);
            }
            nk_end(program.gui_context);

            if (nk_begin(program.gui_context, "Scene - Editor", nk_rect(0, program.h-120, program.w, 120), nk_flags))
            {
                {
                    nk_layout_row_dynamic(program.gui_context, 0, 3);

                    if (nk_button_label(program.gui_context, "Load Sponza"))
                    {
                        free_scene(program.scene);
                        load_test_scene(0, &program.scene);
                        program.cam.pos[0] = 0.0f;
                        program.cam.pos[1] = 1.0f;
                        program.cam.pos[2] = 0.0f;
                        program.cam.pitch = 0.0f;
                        program.cam.yaw = PI/2.0f;
                    }

                    if (nk_button_label(program.gui_context, "Load Suntemple"))
                    {
                        free_scene(program.scene);
                        load_test_scene(1, &program.scene);

                        // glm_vec3_copy((vec3){ 0.577642f, -0.921296f, -28.718777f }, program.cam.pos);
                        // program.cam.pitch = 0.0f;
                        // program.cam.yaw = PI;

                        glm_vec3_copy((vec3){ 3.123726, 0.563292, -53.807980 }, program.cam.pos);
                        program.cam.pitch = 0.095518f;
                        program.cam.yaw = 3.582217f;
                    }

                    if (nk_button_label(program.gui_context, "Load Lost Empire"))
                    {
                        free_scene(program.scene);
                        load_test_scene(2, &program.scene);
                        glm_vec3_copy((vec3){ -13.135565f, 20.071218f, -68.319450f }, program.cam.pos);
                        program.cam.pitch = 0.0f;
                        program.cam.yaw = PI/2.0f;
                    }
                    
                    nk_layout_row_dynamic(program.gui_context, 0, 4);
                    if (nk_button_label(program.gui_context, "Delete all point lights"))
                    {
                        free_array(&program.point_lights);

                        // Create new empty array
                        program.point_lights = create_array(10 * sizeof(PointLight));
                    }

                    if (nk_button_label(program.gui_context, "Delete all area lights"))
                    {
                        free_array(&program.area_lights);

                        // Create new empty array
                        program.area_lights = create_array(10 * sizeof(AreaLight));
                    }

                    if (program.is_clustered_shading_enabled)
                    {
                        nk_label(program.gui_context, "", 0);
                    }
                    
                    const char* label_a = "Disable Clustered Shading";
                    const char* label_b = "Enable Clustered Shading";
                    if (nk_button_label(program.gui_context, program.is_clustered_shading_enabled ? label_a : label_b))
                    {
                        program.is_clustered_shading_enabled = !program.is_clustered_shading_enabled;
                        reload_shaders(1);
                    }

                    static int input_number = CLUSTER_DEFAULT_MAX_LIGHTS;
                    nk_layout_row_dynamic(program.gui_context, 25, 3);
                    nk_label(program.gui_context, "Max lights per cluster:", NK_TEXT_LEFT);

                    // Input field
                    static char buffer[64] = { 0 };
                    snprintf(buffer, sizeof(buffer), "%d", input_number);
                    if (nk_edit_string_zero_terminated(program.gui_context, NK_EDIT_FIELD, buffer, sizeof(buffer), nk_filter_decimal))
                    {
                        input_number = atoi(buffer);
                    }

                    // OK button to submit new max lights and recompile shaders
                    if (nk_button_label(program.gui_context, "Submit (Recompile)"))
                    {
                        program.max_lights_per_cluster = input_number;
                        init_empty_cluster_grid();
                        reload_shaders(0);
                    }
                }
            }
            nk_end(program.gui_context);
        }

#endif
        // Render Scene
        {
            draw_gltf_scene(&program.scene);

#ifndef DISABLE_GUI
            // Render Nuklear GUI
            nk_glfw3_render(&program.gui_glfw, NK_ANTI_ALIASING_ON, NUKLEAR_MAX_VERTEX_BUFFER, NUKLEAR_MAX_ELEMENT_BUFFER);

            // Reset OpenGL state since nuklear fucks it up
            reset_opengl_render_state();
#endif
        }

        glfwSwapBuffers(program.window);
        program.frame_counter++;
        
    }

    // TODO: Prolly should clean up the buffers for no reason if I want to....
    
    nk_glfw3_shutdown(&program.gui_glfw);
    glfwDestroyWindow(program.window);
    glfwTerminate();
    
    return 0;
}
