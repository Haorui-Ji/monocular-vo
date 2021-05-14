//
// Created by jihaorui on 5/2/21.
//

#pragma once
#ifndef MYSLAM_FRONTEND_H
#define MYSLAM_FRONTEND_H

#include <opencv2/features2d.hpp>

#include "myslam/common_include.h"
#include "myslam/frame.h"
#include "myslam/map.h"

namespace myslam {

class Backend;
class Viewer;

enum class FrontendStatus { BLANK, INITING, TRACKING_GOOD, TRACKING_BAD, LOST };

/**
* 前端
* 估计当前帧Pose，在满足关键帧条件时向地图加入关键帧并触发优化
*/
class Frontend {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Frontend> Ptr;

    Frontend();

    /// 外部接口，添加一个帧并计算其定位结果
    bool AddFrame(Frame::Ptr frame);

    /// Set函数
    void SetMap(Map::Ptr map) { map_ = map; }

    void SetBackend(std::shared_ptr<Backend> backend) { backend_ = backend; }

    FrontendStatus GetStatus() const { return status_; }

    void SetCamera(Camera::Ptr camera) {
        camera_ = camera;
    }

private:
    /**
     * Try init the frontend with stereo images saved in current_frame_
     * @return true if success
     */
    bool MonoInit();

    /**
     * Detect features in in current_frame_
     * keypoints will be saved in current_frame_
     * @return
     */
    int DetectFeatures();

    /**
     * Track in normal mode
     * @return true if success
     */
    bool Track();

    /**
     * Reset when lost
     * @return true if success
     */
    bool Reset();

    /**
     * Track with last frame
     * @return num of tracked points
     */
    int TrackLastFrame();

    int TrackRefFrame();

    /**
     * estimate current frame's pose
     * @return num of inliers
     */
    int EstimateCurrentPoseByG2O();

    int EstimateMotionByEpipolarGeometry();

    int EstimateCurrentPoseByPNP();

    /**
     * set current frame as a keyframe and insert it into backend
     * @return true if success
     */
    bool InsertKeyframe();

    bool CheckConditionForAddKeyFrame();

    /**
     * Build the initial map with single image
     * @return true if succeed
     */
    bool BuildInitMap();

    /**
     * Triangulate the 2D points in current frame
     * @return num of triangulated points
     */
    int TriangulateNewPoints();

    /**
     * Set the features in keyframe as new observation of the map points
     */
    void SetObservationsForKeyFrame();

    bool IsGoodToInit();

    // data
    FrontendStatus status_ = FrontendStatus::BLANK;

    Frame::Ptr current_frame_ = nullptr;        // 当前帧
    Frame::Ptr last_frame_ = nullptr;           // 上一帧
    Frame::Ptr reference_frame_ = nullptr;      // 参考帧
    Camera::Ptr camera_ = nullptr;              // 相机

    Map::Ptr map_ = nullptr;
    std::shared_ptr<Backend> backend_ = nullptr;

    cv::Mat relative_motion_;  // 当前帧与上一帧的相对运动，用于估计当前帧 pose 初值

    int tracking_inliers_ = 0;  // inliers, used for testing new keyframes

    // params
    int num_features_tracking_ = 50;
    int num_features_tracking_bad_ = 5;
    int num_features_needed_for_keyframe_ = 100;

    // utilities
    cv::Ptr<cv::ORB> orb_;  // feature detector in opencv
};

}  // namespace myslam

#endif  // MYSLAM_FRONTEND_H

