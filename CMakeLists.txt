cmake_minimum_required(VERSION 3.10)
project(ripple_gl C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -O2 -lm")

find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)

add_executable(ripple
    src/main.c
)

target_include_directories(ripple PRIVATE ${OPENGL_INCLUDE_DIR} ${GLEW_INCLUDE_DIRS})
# Linking - assume GLEW::GLEW and glfw are available on system. If your system provides different targets adjust accordingly.
target_link_libraries(ripple PRIVATE ${OPENGL_gl_LIBRARY} GLEW::GLEW glfw)

