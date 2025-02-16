[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/yfSNuVM-)

![](_ideating/progress_screenshots/19dec-100point_lights-shinyUntitled.png)

TODO: at some point maybe implement statistics from GL_ARB_pipeline_statistics_query
Also maybe wrap malloc with error checking since I never check for null pointer after malloc cuz fork that

Current progress:
Loading GLTF scenes, working PBR metallic roughness and emissives. Gamma correcting only color textures: Using `GL_SRGB8_ALPHA8` for sRGB to linear conversion when loading base_color_texture emissive_texture. Linear to sRGB conversion of final vertex output color done in shader.

![](_ideating/progress_screenshots/22dec-lostworlds-test2.PNG)


Correct alpha blending. Currently only sorting drawcalls by model matrix which doesn't work for gltf files that batch seperate objects into one primitive like the sun temple blender export.
![](_ideating/progress_screenshots/22dec-blendtest.PNG)

TODO: Need Order Independent Transparency instead of current sorting method. The trees below are in the same draw call so order draw calls doesn't solve the artefact below.
![](_ideating/progress_screenshots/22dec-seperate-transparent-pass-but-still-needs-sorting.PNG)


Early november stuff screenshots....
![](_ideating/progress_screenshots/13nov-added-ambient-0.PNG)

Emissive textures shown working here:
![](_ideating/progress_screenshots/13nov-helmet-pbr.PNG)

Vertex normals:
![](_ideating/progress_screenshots/9nov2024-gltf_with_normals.PNG)


![](_ideating/progress_screenshots/11nov-basecolor-texture-not-properly-working-initially.PNG)