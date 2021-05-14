//
// Created by jihaorui on 5/1/21.
//

#pragma once

#ifndef MYSLAM_FEATURE_H
#define MYSLAM_FEATURE_H

#include <memory>
#include <opencv2/features2d.hpp>
#include "myslam/common_include.h"

namespace myslam {

class Frame;
class MapPoint;

/**
 * 2D 特征点
 * 在三角化之后会被关联一个地图点
 */
class Feature
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Feature> Ptr;

    std::weak_ptr<Frame> frame_;            // 持有该 feature 的 frame
    cv::KeyPoint position_;                 // 关键点位置
    cv::Mat descriptor_;                    // 关键点描述子
    std::weak_ptr<MapPoint> map_point_;     // 关联地图点

    bool is_inlier_ = false;                // 是否为内点

    bool associate_new_map_point_ = false;

public:
    Feature() {}

    Feature(std::shared_ptr<Frame> frame, const cv::KeyPoint &kp)
            : frame_(frame), position_(kp){}

    Feature(std::shared_ptr<Frame> frame, const cv::KeyPoint &kp, const cv::Mat &descriptor)
            : frame_(frame), position_(kp), descriptor_(descriptor) {}
};
}  // namespace myslam

#endif  // MYSLAM_FEATURE_H

