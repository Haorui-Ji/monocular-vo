//
// Created by jihaorui on 5/1/21.
//

#pragma once
#ifndef MYSLAM_MAPPOINT_H
#define MYSLAM_MAPPOINT_H

#include "myslam/common_include.h"

namespace myslam
{

class Frame;

class Feature;

/**
 * 路标点类
 * 特征点在三角化之后形成路标点
 */
class MapPoint
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<MapPoint> Ptr;
    unsigned long id_ = 0;              // ID
    bool is_outlier_ = false;
    cv::Point3f pos_;                   // Position in world
    int observed_times_ = 0;            // being observed by feature matching algo.
    std::list<std::weak_ptr<Feature>> observations_;

    MapPoint() {}

    MapPoint(long id, cv::Point3f position);

    cv::Point3f Pos() {
        return pos_;
    }

    void SetPos(const cv::Point3f &pos) {
        pos_ = pos;
    };

    void AddObservation(std::shared_ptr<Feature> feature) {
        observations_.push_back(feature);
        observed_times_++;
    }

    void RemoveObservation(std::shared_ptr<Feature> feat);

    std::list<std::weak_ptr<Feature>> GetObs() {
        return observations_;
    }

    // factory function
    static MapPoint::Ptr CreateNewMappoint();
};
}  // namespace myslam

#endif  // MYSLAM_MAPPOINT_H

