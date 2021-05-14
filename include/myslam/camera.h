//
// Created by jihaorui on 5/1/21.
//

#pragma once

#ifndef MYSLAM_CAMERA_H
#define MYSLAM_CAMERA_H

#include "myslam/common_include.h"

namespace myslam
{

// Pinhole camera model
class Camera
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Camera> Ptr;

    double fx_ = 0, fy_ = 0, cx_ = 0, cy_ = 0;
    cv::Mat K_;

    Camera();

    Camera(double fx, double fy, double cx, double cy)
            : fx_(fx), fy_(fy), cx_(cx), cy_(cy) {
        K_ = (cv::Mat_<double>(3, 3) << fx_, 0, cx_, 0, fy_, cy_, 0, 0, 1);
    }

    Camera(cv::Mat K)
    {
        fx_ = K.at<double>(0, 0);
        fy_ = K.at<double>(1, 1);
        cx_ = K.at<double>(0, 2);
        cy_ = K.at<double>(1, 2);
        K_ = K;
    }

    // coordinate transform: world, camera, pixel
    cv::Point3f world2camera(const cv::Point3f &p, const cv::Mat &T_c_w);

    cv::Point3f camera2world(const cv::Point3f &p, const cv::Mat &T_c_w);

    cv::Point3f pixel2camera(const cv::Point2f &p, double depth = 1);

    cv::Point2f camera2pixel(const cv::Point3f &p);

    cv::Point2f world2pixel(const cv::Point3f &p, const cv::Mat &T_c_w);

    cv::Point3f pixel2world (const cv::Point2f &p, const cv::Mat &T_c_w, double depth = 1);
};

}
#endif  // MYSLAM_CAMERA_H

