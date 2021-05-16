//
// Created by jihaorui on 5/8/21.
//

#pragma once
#ifndef MYSLAM_VISUAL_ODOMETRY_H
#define MYSLAM_VISUAL_ODOMETRY_H

#include "myslam/backend.h"
#include "myslam/common_include.h"
#include "myslam/dataset.h"
#include "myslam/frontend.h"
#include "myslam/config.h"

namespace myslam
{

/**
 * VO 对外接口
 */
class VisualOdometry
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<VisualOdometry> Ptr;

    /// constructor with config file
    VisualOdometry(std::string &config_path);

    /**
     * do initialization things before run
     * @return true if success
     */
    bool Init();

    /**
     * start vo in the dataset
     */
    void Run();

    /**
     * Make a step forward in dataset
     */
    bool Step();

    /// 获取数据集信息
    Dataset::Ptr GetDataset() const { return dataset_; }

    /// 获取前端状态
    FrontendStatus GetFrontendStatus() const { return frontend_->GetStatus(); }

    /// 获取每一帧的位姿结果
    vector<cv::Mat> GetCamPoseHistory() const { return cam_pose_history_; }

    /// 将相机位姿写入文本文档
    void WritePoseToFile(const string filename);

    cv::Mat traj_ = Mat::zeros(1000, 1000, CV_8UC3);

private:
    bool inited_ = false;
    std::string config_file_path_;

    Frontend::Ptr frontend_ = nullptr;
    Backend::Ptr backend_ = nullptr;
    Map::Ptr map_ = nullptr;

    // dataset
    Dataset::Ptr dataset_ = nullptr;

    // camera pose vector
    vector<cv::Mat> cam_pose_history_;
};

}  // namespace myslam

#endif  // MYSLAM_VISUAL_ODOMETRY_H
