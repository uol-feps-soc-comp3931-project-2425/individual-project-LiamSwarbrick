
// Defined anti-clockwise, these are n-gons, not triangle meshes
// This makes it problematic to actually render the light sources when they are concave
// since that would require triangulation, so instead I'm just using convex polygons,
// except the star, where I've hardprogrammed the triangle mesh to avoid having to implement
// an ear-clipping algorithm for concave ngon triangulation.

const float triangle_points[12] = {
    -0.5f, -0.5f, 0.0f, 1.0f,
     0.5f, -0.5f, 0.0f, 1.0f,
     0.0f,  0.5f, 0.0f, 1.0f,
};

const float quad_points[16] = {
    -0.5f, -0.5f, 0.0f, 1.0f,
     0.5f, -0.5f, 0.0f, 1.0f,
     0.5f,  0.5f, 0.0f, 1.0f,
    -0.5f,  0.5f, 0.0f, 1.0f,
};

const float pentagon_points[20] = {
    -0.3f, -0.5f, 0.0f, 1.0f,
     0.3f, -0.5f, 0.0f, 1.0f,
     0.5f,  0.3f, 0.0f, 1.0f,
     0.0f,  0.8f, 0.0f, 1.0f,
    -0.5f,  0.3f, 0.0f, 1.0f,
};

const float star_points[40] = {
    -0.3536f, -0.3536f, 0.0f, 1.0f,
    -0.0313f, -0.1975f, 0.0f, 1.0f,
     0.2270f, -0.4455f, 0.0f, 1.0f,
     0.1782f, -0.0908f, 0.0f, 1.0f,
     0.4939f,  0.0782f, 0.0f, 1.0f,
     0.1414f,  0.1414f, 0.0f, 1.0f,
     0.0782f,  0.4939f, 0.0f, 1.0f,
    -0.0908f,  0.1782f, 0.0f, 1.0f,
    -0.4455f,  0.2270f, 0.0f, 1.0f,
    -0.1975f, -0.0313f, 0.0f, 1.0f,
};

const unsigned int star_indices[24] = {
    // star tips
    9,0,1, 1,2,3, 3,4,5, 5,6,7, 7,8,9,

    // inner part
    7,9,1, 7,1,3, 7,3,5,
};
