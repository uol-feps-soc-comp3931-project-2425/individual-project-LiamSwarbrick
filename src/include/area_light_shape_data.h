#ifndef AREA_LIGHT_SHAPE_DATA_H
#define AREA_LIGHT_SHAPE_DATA_H

// TODO: Pressing 1-9 keys should scale the last area light in the array.

// Defined anti-clockwise, these are n-gons, not triangle meshes
// This makes it problematic to actually render the light sources when they are concave
// since that would require triangulation, so instead I'm just using convex polygons,
// except the star, where I've hardprogrammed the triangle mesh to avoid having to implement
// an ear-clipping algorithm for concave ngon triangulation.

const float triangle_points[] = {
    -0.5f, -0.5f, 0.0f, 1.0f,
     0.5f, -0.5f, 0.0f, 1.0f,
     0.0f,  0.5f, 0.0f, 1.0f,
};

const float quad_points[] = {
    -0.5f, -0.5f, 0.0f, 1.0f,
     0.5f, -0.5f, 0.0f, 1.0f,
     0.5f,  0.5f, 0.0f, 1.0f,
    -0.5f,  0.5f, 0.0f, 1.0f,
};

const float pentagon_points[] = {
    -0.3f, -0.5f, 0.0f, 1.0f,
     0.3f, -0.5f, 0.0f, 1.0f,
     0.5f,  0.3f, 0.0f, 1.0f,
     0.0f,  0.8f, 0.0f, 1.0f,
    -0.5f,  0.3f, 0.0f, 1.0f,
};

const float star_points[] = {
    -0.3536f, -0.3536f, 0.0f, 1.0f,  // Vertex 0: Outer, angle -135° (bottom left)
    -0.0313f, -0.1975f, 0.0f, 1.0f,  // Vertex 1: Inner, angle -99°
     0.2270f, -0.4455f, 0.0f, 1.0f,  // Vertex 2: Outer, angle -63°
     0.1782f, -0.0908f, 0.0f, 1.0f,  // Vertex 3: Inner, angle -27°
     0.4939f,  0.0782f, 0.0f, 1.0f,  // Vertex 4: Outer, angle 9°
     0.1414f,  0.1414f, 0.0f, 1.0f,  // Vertex 5: Inner, angle 45°
     0.0782f,  0.4939f, 0.0f, 1.0f,  // Vertex 6: Outer, angle 81°
    -0.0908f,  0.1782f, 0.0f, 1.0f,  // Vertex 7: Inner, angle 117°,
    -0.4455f,  0.2270f, 0.0f, 1.0f,  // Vertex 8: Outer, angle 153°
    -0.1975f, -0.0313f, 0.0f, 1.0f,  // Vertex 9: Inner, angle 189° (equivalent to -171°)
};

const unsigned int star_indices[] = {
    // star tips
    9,0,1, 1,2,3, 3,4,5, 5,6,7, 7,8,9,

    // inner part
    7,9,1, 7,1,3, 7,3,5,

    // 0, 2, 4,
    // 0, 4, 6,
    // 0, 6, 8,
    // 0, 8, 1,
    // 8, 1, 3,
    // 8, 3, 5,
    // 8, 5, 7,
    // 8, 7, 0
};

#endif  // AREA_LIGHT_SHAPE_DATA_H
