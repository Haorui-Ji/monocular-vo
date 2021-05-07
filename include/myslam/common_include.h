#pragma once
#ifndef MYSLAM_COMMON_INCLUDE_H
#define MYSLAM_COMMON_INCLUDE_H

// std
#include <atomic>
#include <condition_variable>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <typeinfo>
#include <unordered_map>
#include <vector>

// eigen
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>

// cv
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
using cv::Mat;

// glog
#include <glog/logging.h>

// pcl
#include <pcl/point_types.h>
#include <pcl/registration/ndt.h>
#include <pcl/filters/approximate_voxel_grid.h>

// ceres
#include <ceres/ceres.h>

using namespace std;

#endif //MYSLAM_COMMON_INCLUDE_H