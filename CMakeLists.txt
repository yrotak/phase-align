cmake_minimum_required(VERSION 3.0)

project(phase-align
    VERSION 1.0
    LANGUAGES CXX
    DESCRIPTION "Audio phase alignment tool"
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_compile_options(-O3 -march=native)
file(GLOB_RECURSE SRC src/*.cpp src/*.c)
file(GLOB_RECURSE HEADERS src/*.hpp src/*.h src/*.inl)

find_package(SDL2 REQUIRED)

if(NOT SDL2_INCLUDE_DIRS)
    set(SDL2_INCLUDE_DIRS $ENV{SDL2_INCLUDE_DIR})
endif()
if(NOT SDL2_LIBRARIES)
    set(SDL2_LIBRARIES $ENV{SDL2_LIBRARY})
endif()
include_directories(${SDL2_INCLUDE_DIRS})

set (CMAKE_CXX_LINK_EXECUTABLE "${CMAKE_CXX_LINK_EXECUTABLE} -ldl -lGL -lpthread") 

add_executable(phase-align ${SRC} ${HEADERS})

if(WIN32)
    target_link_libraries(phase-align ${SDL2_LIBRARIES} opengl32)
else()
    target_link_libraries(phase-align ${SDL2_LIBRARIES} dl GL pthread)
endif()