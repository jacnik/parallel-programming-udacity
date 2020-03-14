cmake_minimum_required(VERSION 2.8.12)

project(demo)

# SET("OpenCV_DIR" "/usr/local/include/opencv2/")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("/usr/local/include/opencv2/")


#MESSAGE(STATUS "Include dirs ${OpenCV_INCLUDE_DIRS}")
#MESSAGE(STATUS "LINK LIBRARIES ${OpenCV_LIBS}")

add_executable(demo
    demo.cpp
    /usr/local/include/opencv2/viz)

target_link_libraries(demo ${OpenCV_LIBS})
