#!/bin/bash

exe_name=a.out
gcc_args="-g -Wall -Wextra"
src_dirs="external/*.c src/*.c"
include_dirs="-Iexternal -Isrc/include"
link_args="-lglfw -lGL -lm -lpthread -ldl"

gcc $gcc_args $src_dirs $include_dirs -o $exe_name $link_args
