//
// Created by jihaorui on 5/4/21.
//

//
// Created by jihaorui on 2/25/21.
//

#ifndef MYSLAM_UTILS_H
#define MYSLAM_UTILS_H

#include "myslam/common_include.h"

namespace myslam
{

    // ---------------- datatype conversion ----------------
    cv::Mat point3f_to_mat3x1(const cv::Point3f &p);

    cv::Mat point3f_to_mat4x1(const cv::Point3f &p);

    cv::Point3f mat3x1_to_point3f(const cv::Mat &p);

    cv::Mat point2f_to_mat2x1(const cv::Point2f &p);

    cv::Mat getZerosMat(int rows, int cols, int type);
    // ----------------------------------------------------------

    cv::Mat convertRt2T_3x4(const cv::Mat &R, const cv::Mat &t);

    cv::Mat convertRt2T(const cv::Mat &R, const cv::Mat &t);

    void getRtFromT(const cv::Mat &T, cv::Mat &R, cv::Mat &t);

    cv::Mat getPosFromT(const cv::Mat &T);

    cv::Point3f transCoord(const cv::Point3f &p, const cv::Mat &R, const cv::Mat &t);

    cv::Point3f transCoord(const cv::Point3f &p, const cv::Mat &T4x4);

    double calcAngleBetweenTwoVectors(const cv::Mat &vec1, const cv::Mat &vec2);

}

#endif //MYSLAM_UTILS_H

