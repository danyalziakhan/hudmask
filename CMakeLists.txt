cmake_minimum_required(VERSION 3.20)
project(hudmask LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

add_library(hudmask SHARED src/hudmask.cpp src/hudmask.rc)
target_include_directories(hudmask PRIVATE deps/reshade/include deps/imgui)
target_compile_definitions(hudmask PRIVATE UNICODE _UNICODE _CRT_SECURE_NO_WARNINGS)
target_compile_options(hudmask PRIVATE /W3 /utf-8)
set_target_properties(hudmask PROPERTIES SUFFIX ".addon64")
