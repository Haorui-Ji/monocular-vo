//
// Created by jihaorui on 5/1/21.
//

#include "myslam/frame.h"
#include "myslam/utils.h"

namespace myslam
{

Frame::Frame(long id, double time_stamp, const cv::Mat &pose, const cv::Mat &rgb, const cv::Mat &depth)
        : id_(id), time_stamp_(time_stamp), pose_(pose), rgb_img_(rgb), depth_img_(depth) {}

Frame::Ptr Frame::CreateFrame() {
    static long factory_id = 0;
    Frame::Ptr new_frame(new Frame);
    new_frame->id_ = factory_id++;
    return new_frame;
}

void Frame::SetKeyFrame() {
    static long keyframe_factory_id = 0;
    is_keyframe_ = true;
    keyframe_id_ = keyframe_factory_id++;
}

bool Frame::IsInFrame(const cv::Point3f &p_world)
{
    cv::Point3f p_cam = transCoord(p_world, pose_);
    if (p_cam.z < 0) return false;

    cv::Point2f pixel = camera_->camera2pixel(p_cam);
    return pixel.x > 0 && pixel.y > 0 && pixel.x < rgb_img_.cols && pixel.y < rgb_img_.rows;
}

}