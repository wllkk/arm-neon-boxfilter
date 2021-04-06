cmake_minimum_required(VERSION 3.5)

project(BOXFILTER)

#set(CMAKE_BUILD_TYPE Release)

add_compile_options(-Wall -std=c++11 -fopenmp -O3)

find_package(OpenCV)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/src)
aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC_LIST)


set(SOURCE_FILES test.cpp)

add_executable(boxfilter ${SOURCE_FILES} ${SRC_LIST})

target_link_libraries(boxfilter ${OpenCV_LIBS})

