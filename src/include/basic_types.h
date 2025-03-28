#ifndef BASIC_TYPES_H
#define BASIC_TYPES_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
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

float randomf();
float rng_rangef(float min, float max);

void hsv_to_rgb(float h, float s, float v, float out_rgb[3]);
void rgb_to_hsv(float r, float g, float b, float out_hsv[3]);

char* malloc_strcat(const char* a, const char* b);
char* load_text_file(const char* filename);


typedef struct DynamicArray
{
    void* data_buffer;
    size_t used_size;
    size_t capacity;
}
DynamicArray;

DynamicArray create_array(size_t starting_capacity);
void free_array(DynamicArray* arr);
void* push_size(DynamicArray* dest, size_t element_size, size_t count);
void* push_element_copy(DynamicArray* dest, size_t element_size, void* src);
void* get_element(DynamicArray* arr, size_t element_size, size_t index);
size_t array_length(DynamicArray* arr, size_t element_size);

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
    
    free_array(&arr);
}
*/

#endif  // BASIC_TYPES_H
