#ifndef BASIC_TYPES_H
#define BASIC_TYPES_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

typedef int8_t   s8;
typedef int16_t  s16;
typedef int32_t  s32;
typedef int64_t  s64;

typedef uint8_t  u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef int8_t  b8;
typedef int16_t b16;
typedef int32_t b32;
typedef int64_t b64;

typedef float    f32;
typedef double   f64;

#define PI  3.14159265358979323846f
#define max(a,b) (a) > (b) ? (a) : (b)
#define min(a,b) (a) < (b) ? (a) : (b)

// Stringification macros needed for sharing macros between C and GLSL
#define xstr(s) str(s)
#define str(s) #s

float
randomf()
{
    // This method needs better seeding
    // static u32 seed = 21258314;
    // u32 xorshift32 = seed++;
    // xorshift32 ^= xorshift32 << 13;
    // xorshift32 ^= xorshift32 >> 17;
    // xorshift32 ^= xorshift32 << 5;
    
    // return (float)xorshift32 / (float)UINT32_MAX; 

    // Basic worse alternative...
    return (float)rand() / (float)RAND_MAX;
}

float
rng_rangef(float min, float max)
{
    return min + (max - min) * randomf();
}

void
hsv_to_rgb(float h, float s, float v, float out_rgb[3])
{
    float r, g, b;

    int i = (int)(h * 6.0f); // Sector of the color wheel (0-5)
    float f = (h * 6.0f) - i; // Fractional part
    float p = v * (1.0f - s);
    float q = v * (1.0f - s * f);
    float t = v * (1.0f - s * (1.0f - f));

    switch (i % 6) {
        case 0: r = v; g = t; b = p; break;
        case 1: r = q; g = v; b = p; break;
        case 2: r = p; g = v; b = t; break;
        case 3: r = p; g = q; b = v; break;
        case 4: r = t; g = p; b = v; break;
        case 5: r = v; g = p; b = q; break;
        default: r = g = b = 0; break; // Should never happen
    }

    out_rgb[0] = r;
    out_rgb[1] = g;
    out_rgb[2] = b;
}


char*
malloc_strcat(const char* a, const char* b)
{
    size_t a_len = strlen(a);
    size_t b_len = strlen(b);
    size_t ab_len = a_len + b_len;

    char* ab = malloc(ab_len + 1);
    if (ab == NULL)
    {
        return NULL;
    }

    memcpy(ab, a, a_len);
    memcpy(ab + a_len, b, b_len);
    ab[ab_len] = '\0';

    return ab;
}

char*
load_text_file(const char* filename)
{
    FILE* file = fopen(filename, "rb");
    if (file == NULL)
    {
        return NULL;
    }

    fseek(file, 0, SEEK_END);
    size_t file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    char* buffer = calloc(1, file_size + 1);
    if (buffer == NULL)
    {
        fclose(file);
        return NULL;
    }

    fread(buffer, 1, file_size, file);
    fclose(file);

    buffer[file_size] = '\0';
    
    return buffer;
}

typedef struct DynamicArray
{
    void* data_buffer;
    size_t used_size;
    size_t capacity;
}
DynamicArray;

DynamicArray
create_array(size_t starting_capacity)
{
    // assert(starting_capacity > 0 && "Dynamic_Array must be initialized with > 0 starting_capacity");
    if (starting_capacity == 0)
    {
        starting_capacity = 32;  // Default to 32 bytes  
    }

    DynamicArray arr;
    arr.data_buffer = malloc(starting_capacity);
    arr.used_size = 0;
    arr.capacity = starting_capacity;
    return arr;
}

void
free_array(DynamicArray* arr)
{
    if (arr->data_buffer)
    {
        free(arr->data_buffer);
    }
}

void*
push_size(DynamicArray* dest, size_t element_size, size_t count)
{
    // Double data buffer length when capacity is overran
    size_t new_used_size = dest->used_size + element_size * count;
    while (new_used_size > dest->capacity)
    {
        dest->capacity *= 2;
        dest->data_buffer = realloc(dest->data_buffer, dest->capacity);
        if (dest->data_buffer == NULL)
        {
            fprintf(stderr, "Dynamic_Array Reallocation failed!\n");
            exit(1);
        }
    }
    
    void* new_elements = dest->data_buffer + dest->used_size;
    dest->used_size = new_used_size;

    // Return a pointer to the first of the newly pushed elements in the array
    return new_elements;
}

void*
push_element_copy(DynamicArray* dest, size_t element_size, void* src)
{
    assert(src != NULL);

    void* new_buffer = push_size(dest, element_size, 1);
    memcpy(new_buffer, src, element_size);

    // Return pointer to newly added element in the array
    return new_buffer;
}

void*
get_element(DynamicArray* arr, size_t element_size, size_t index)
{
    assert(element_size * (index+1) <= arr->used_size);

    return (char*)arr->data_buffer + (element_size * index);
}

size_t
array_length(DynamicArray* arr, size_t element_size)
{
    assert(element_size != 0);

    return arr->used_size / element_size;
}

/*
typedef struct Bean
{
    int a;
    float b;
    char c[4];
}
Bean;

void
test_dynamic_array()
{
    Dynamic_Array arr = create_array(sizeof(Bean));

    for (int i = 0; i < 50; ++i)
    {
        Bean b1 = { 1, i, "hi!" };
        push_element_copy(&arr, sizeof(Bean), &b1);
    }


    for (int i = arr.used_size / sizeof(Bean)-1; i >= 0; --i)
    {
        Bean* b = get_element(&arr, sizeof(Bean), i);
        printf("%i: %d %f %s\n", i, b->a, b->b, b->c);
    }
}
*/

#endif  // BASIC_TYPES_H
