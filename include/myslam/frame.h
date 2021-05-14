//
// Created by jihaorui on 5/1/21.
//

#pragma once

#ifndef MYSLAM_FRAME_H
#define MYSLAM_FRAME_H

#include "myslam/camera.h"
#include "myslam/common_include.h"

namespace myslam
{

// forward declare
class MapPoint;
class Feature;

/**
 * 帧
 * 每一帧分配独立id，关键帧分配关键帧ID
 */
class Frame {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Frame> Ptr;

    unsigned long id_ = 0;           // id of current frame
    unsigned long keyframe_id_ = 0;  // id of key frame
    bool is_keyframe_ = false;       // 是否为关键帧
    double time_stamp_;              // 时间戳，暂不使用
    cv::Mat pose_;                   // Tcw 形式 Pose
    cv::Mat rgb_img_, depth_img_;    // rgb 和深度图

    // Pinhole RGB Camera model
    Camera::Ptr camera_;

    // extracted features from image
    vector<std::shared_ptr<Feature>> features_;

    vector<cv::DMatch> matches_with_ref_frame_;         // matches with reference frame
    vector<cv::DMatch> matches_with_last_frame_;        // matches with last frame

public:  // data members
    Frame() {}

    Frame(long id, double time_stamp, const cv::Mat &pose, const cv::Mat &rgb, const cv::Mat &depth);

    // set and get pose
    cv::Mat Pose() {
        return pose_;
    }

    void SetPose(const cv::Mat &pose) {
        pose_ = pose;
    }

    /// 设置关键帧并分配并键帧id
    void SetKeyFrame();

    /// 工厂构建模式，分配id
    static std::shared_ptr<Frame> CreateFrame();

    /// check if a point is in this frame
    bool IsInFrame(const cv::Point3f &p_world);
};

}  // namespace myslam

#endif  // MYSLAM_FRAME_H
