//
// Created by jihaorui on 5/7/21.
//

#ifndef MYSLAM_ALGORITHM_H
#define MYSLAM_ALGORITHM_H

#include "myslam/common_include.h"
#include "myslam/feature.h"
#include "myslam/camera.h"

namespace myslam
{

void MatchFeatures(
        const vector<std::shared_ptr<Feature>> &features_1,
        const vector<std::shared_ptr<Feature>> &features_2,
        vector<cv::DMatch> &matches);

void Triangulation(
        const vector<cv::Point2f> &inlier_pts_in_img1,
        const vector<cv::Point2f> &inlier_pts_in_img2,
        const Camera::Ptr &camera,
        const cv::Mat &R21, const cv::Mat &t21,
        vector<cv::Point3f> &pts3d_in_cam1);

vector<bool> CheckGoodTriangulationResult(
        const cv::Mat &pose_1,
        const cv::Mat &pose_2,
        const Camera::Ptr &camera,
        const vector<cv::Point3f> pts3d_in_cam1,
        const vector<cv::Point2f> inlier_pts_in_img_1,
        const vector<cv::Point2f> inlier_pts_in_img_2);

void FindInliersByEpipolar(
        const cv::Mat &pose_1,
        const cv::Mat &pose_2,
        const Camera::Ptr &camera,
        const vector<std::shared_ptr<Feature>> &features_1,
        const vector<std::shared_ptr<Feature>> &features_2,
        vector<cv::DMatch> &matches);

}  // namespace myslam

#endif  // MYSLAM_ALGORITHM_H
