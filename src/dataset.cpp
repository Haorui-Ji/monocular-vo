//
// Created by jihaorui on 5/1/21.
//

#include "myslam/dataset.h"
#include "myslam/frame.h"

#include <boost/format.hpp>
#include <fstream>
#include <opencv2/opencv.hpp>
using namespace std;

namespace myslam {

Dataset::Dataset(const std::string& dataset_path)
        : dataset_path_(dataset_path) {}

bool Dataset::Init() {

    // read time stamp
    ifstream time_stream(dataset_path_ + "/times.txt");
    if (!time_stream) {
        LOG(ERROR) << "cannot find " << dataset_path_ << "/times.txt!";
        return false;
    }
    double t;
    while(time_stream >> t) {
        times_.push_back(t);
    }

    // read camera intrinsics and extrinsics
    ifstream fin(dataset_path_ + "/calib.txt");
    if (!fin) {
        LOG(ERROR) << "cannot find " << dataset_path_ << "/calib.txt!";
        return false;
    }

    for (int i = 0; i < 4; ++i) {
        char camera_name[3];
        for (int k = 0; k < 3; ++k) {
            fin >> camera_name[k];
        }
        double projection_data[12];
        for (int k = 0; k < 12; ++k) {
            fin >> projection_data[k];
        }
        cv::Mat K = (cv::Mat_<double>(3, 3) <<
                projection_data[0], projection_data[1], projection_data[2],
                projection_data[4], projection_data[5], projection_data[6],
                projection_data[8], projection_data[9], projection_data[10]);
        cv::Mat t = (cv::Mat_<double>(3, 1) <<
                projection_data[3], projection_data[7], projection_data[11]);
        t = K.inv() * t;
//        K = K * 0.5;
        Camera::Ptr new_camera(new Camera(K));
        cameras_.push_back(new_camera);
        LOG(INFO) << "Camera " << i << " extrinsics: " << t.t();
    }
    fin.close();
    current_image_index_ = 0;
    return true;
}

Frame::Ptr Dataset::NextFrame() {
    boost::format fmt("%s/image_%d/%06d.png");
    cv::Mat image;

    // read images
    image = cv::imread((fmt % dataset_path_ % 2 % current_image_index_).str(),
                       cv::IMREAD_COLOR);

    /*
     * Currently we're ignoring the input of depth image
     * TODO in the future
     */

    if (image.data == nullptr) {
        LOG(WARNING) << "cannot find images at index " << current_image_index_;
        return nullptr;
    }

//    cv::Mat image_resized;
//    cv::resize(image, image_resized, cv::Size(), 0.5, 0.5,
//               cv::INTER_NEAREST);

    auto new_frame = Frame::CreateFrame();
    new_frame->rgb_img_ = image;
    current_image_index_++;
    return new_frame;
}

}  // namespace myslam
