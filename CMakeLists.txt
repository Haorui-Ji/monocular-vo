cmake_minimum_required(VERSION 3.15)
project(myslam)

set(CMAKE_CXX_STANDARD 14)

############### dependencies ######################
# Eigen
include_directories("/usr/include/eigen3")

# OpenCV
find_package(OpenCV 4.0 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Ceres
find_package( Ceres REQUIRED )
include_directories( ${Ceres_INCLUDE_DIRS} )

# PCL
find_package( PCL 1.8 REQUIRED )
include_directories(${PCL_INCLUDE_DIRS})

# pangolin
find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

# G2O

LIST( APPEND CMAKE_MODULE_PATH /home/jihaorui/NewDisk/github/g2o/cmake_modules )
SET( G2O_ROOT /home/jihaorui/NewDisk/github/g2o)
find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIRS})

# glog
find_package(Glog REQUIRED)
include_directories(${GLOG_INCLUDE_DIRS})

set(THIRD_PARTY_LIBS
        ${OpenCV_LIBS}
        ${Ceres_LIBRARIES}
        ${PCL_LIBRARY_DIRS}
        ${Pangolin_LIBRARIES}
        ${GLOG_LIBRARIES}
        g2o_core g2o_stuff g2o_types_sba g2o_solver_csparse g2o_csparse_extension
        pthread
        )

############### source and test ######################
include_directories(${PROJECT_SOURCE_DIR}/include)
add_subdirectory(src)
add_subdirectory(test)

add_executable(myslam main.cpp)
target_link_libraries(myslam myslam_lib)