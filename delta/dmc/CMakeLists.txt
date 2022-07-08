

cmake_minimum_required(VERSION 3.16)


# set the project name
project(dmc LANGUAGES CXX CUDA)
#set(CMAKE_CUDA_FLAGS "-Wall")

set(CMAKE_CXX_STANDARD 14)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package(OpenCV)



# add the executable
add_executable(dmc src/main.cpp src/tree_update.cu include/dmc/update_tree.h)


target_include_directories(dmc PUBLIC
        include
        ${Opencv_INCLUDE_DIRS}
        )

target_link_libraries(dmc
        ${OpenCV_LIBS}
        )
