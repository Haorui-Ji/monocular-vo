//
// Created by jihaorui on 5/1/21.
//

#include "myslam/camera.h"

namespace myslam {

Camera::Camera() {
}

cv::Point3f Camera::world2camera(const cv::Point3f &p_w, const cv::Mat &T_c_w)
{
    cv::Mat p_w_h = (cv::Mat_<double>(4, 1) << p_w.x, p_w.y, p_w.z, 1);
    cv::Mat p_c_h = T_c_w * p_w_h;
    return cv::Point3f(p_c_h.at<double>(0, 0),
                       p_c_h.at<double>(1, 0),
                       p_c_h.at<double>(2, 0));
}

cv::Point3f Camera::camera2world(const cv::Point3f &p_c, const cv::Mat &T_c_w)
{
    cv::Mat R = T_c_w.rowRange(0, 3).colRange(0, 3);
    cv::Mat t = T_c_w.rowRange(0, 3).col(3);

    cv::Mat T_c_w_4x4 = cv::Mat::eye(4, 4, CV_64F);
    R.copyTo(T_c_w.rowRange(0, 3).colRange(0, 3));
    t.copyTo(T_c_w.rowRange(0, 3).col(3));

    cv::Mat p_c_h = (cv::Mat_<double>(4, 1) << p_c.x, p_c.y, p_c.z, 1);
    cv::Mat p_w_h = T_c_w_4x4.inv() * p_c_h;
    return cv::Point3f(p_w_h.at<double>(0, 0),
                       p_w_h.at<double>(1, 0),
                       p_w_h.at<double>(2, 0));
}

cv::Point3f Camera::pixel2camera(const cv::Point2f &p, double depth)
{
    return cv::Point3f(
            depth * (p.x - K_.at<double>(0, 2)) / K_.at<double>(0, 0),
            depth * (p.y - K_.at<double>(1, 2)) / K_.at<double>(1, 1),
            depth);
}

cv::Point2f Camera::camera2pixel(const cv::Point3f &p)
{
    return cv::Point2f(
            K_.at<double>(0, 0) * p.x / p.z + K_.at<double>(0, 2),
            K_.at<double>(1, 1) * p.y / p.z + K_.at<double>(1, 2));
}

cv::Point2f Camera::world2pixel ( const cv::Point3f &p, const cv::Mat &T_c_w)
{
    return camera2pixel ( world2camera(p, T_c_w) );
}

cv::Point3f Camera::pixel2world ( const cv::Point2f &p, double depth, const cv::Mat &T_c_w )
{
    return camera2world ( pixel2camera ( p, depth ), T_c_w );
}

}


