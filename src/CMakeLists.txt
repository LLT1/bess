cmake_minimum_required(VERSION 3.15)
project(BeSS)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)
include_directories(Eigen)

AUX_SOURCE_DIRECTORY(. DIR_SRCS)
ADD_EXECUTABLE(BeSS ${DIR_SRCS})
#
#add_executable(BeSS ${SOURCE_FILES})

#find_package(BeSS REQUIRED)
#target_link_libraries(BESS BeSS::BeSS)



#add_executable(src
#        Algorithm.cpp
#        Algorithm.h
#        Data.cpp
#        Data.h
#        List.cpp
#        List.h
#        main.cpp
#        Metric.cpp
#        Metric.h
#        path.cpp
#        logistic.cpp
#        logistic.h
#        path.h)
