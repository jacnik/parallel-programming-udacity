cmake_minimum_required(VERSION 2.8.12)

project(demo2)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(demo2.out main.cpp)
target_link_libraries(demo2.out ${OpenCV_LIBS})
