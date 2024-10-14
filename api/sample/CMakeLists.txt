cmake_minimum_required(VERSION 3.21)
project(airuntime-sample)

set(CMAKE_CXX_STANDARD 17)

set(SOURCES 
    main.cc
    )
add_executable(airuntime-sample ${SOURCES})

include_directories(../../include)

find_package(glog REQUIRED)

set(LIBS glog::glog)

target_link_libraries(airuntime-sample ${LIBS} airuntime)
