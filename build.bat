@echo off

set exe_name=a
set gcc_args=-g -Wall -Wextra
set src_dirs=external/*.c src/*.c
set include_dirs=-Iexternal -Isrc/include
set link_args=-Lexternal -lglfw3 -lopengl32 -lgdi32

gcc %gcc_args% %src_dirs% %include_dirs% -o %exe_name% %link_args%

@echo on
