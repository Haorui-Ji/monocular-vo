//
// Created by jihaorui on 5/8/21.
//

#include "myslam/visual_odometry.h"
#include "myslam/config.h"
#include "myslam/feature.h"
#include "myslam/mappoint.h"
#include <boost/format.hpp>
#include "myslam/utils.h"

namespace myslam
{

VisualOdometry::VisualOdometry(std::string &config_path)
        : config_file_path_(config_path) {}

bool VisualOdometry::Init()
{
    // read from config file
    if (Config::SetParameterFile(config_file_path_) == false) {
        return false;
    }

    dataset_ =
            Dataset::Ptr(new Dataset(Config::Get<std::string>("dataset_dir")));
    CHECK_EQ(dataset_->Init(), true);

    // create components and links
    frontend_ = Frontend::Ptr(new Frontend);
    backend_ = Backend::Ptr(new Backend);
    map_ = Map::Ptr(new Map);

    frontend_->SetBackend(backend_);
    frontend_->SetMap(map_);
    frontend_->SetCamera(dataset_->GetCamera(2));

    return true;
}

void VisualOdometry::Run() {
    int count = 0;
    while (1) {
        LOG(INFO) << "VO is running on frame " << count;
        if (Step() == false) {
            break;
        }
        count++;
    }

    LOG(INFO) << "VO exit";
}

bool VisualOdometry::Step() {
    Frame::Ptr new_frame = dataset_->NextFrame();
    if (new_frame == nullptr) return false;

//    cv::imshow ( "current frame", new_frame->rgb_img_ );
//    cv::waitKey(0);

    auto t1 = std::chrono::steady_clock::now();
    bool success = frontend_->AddFrame(new_frame);
    auto t2 = std::chrono::steady_clock::now();
    auto time_used =
            std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    LOG(INFO) << "VO cost time: " << time_used.count() << " seconds.";

    cam_pose_history_.push_back(new_frame->Pose().clone().inv());

    /////////////////// Visualize 1 ///////////////////////////////
    cv::Mat traj = Mat::zeros(600, 600, CV_8UC3);
    cv::Mat R, t_vec;
    getRtFromT(new_frame->Pose().inv(), R, t_vec);
    vector<vector<float>> poses = dataset_->GetGroundTruthPose();
    string text  = "Red color: estimated trajectory";
    string text2 = "Blue color: Groundtruth trajectory";

    t_vec.convertTo(t_vec, CV_32F);
    cv::Point2f center = cv::Point2f(int(t_vec.at<float>(0)) + 300, int(t_vec.at<float>(2)) + 100);
    cv::Point2f t_center = cv::Point2f(int(poses[new_frame->id_][3]) + 300, int(poses[new_frame->id_][11]) + 100);
    cv::circle(traj, center, 1, cv::Scalar(0, 0, 255), 2);
    cv::circle(traj, t_center, 1, cv::Scalar(255, 0, 0), 2);
    cv::rectangle(traj, cv::Point2f(10, 30), cv::Point2f(550, 50),  cv::Scalar(0,0,0), cv::FILLED);
    cv::putText(traj, text, cv::Point2f(10, 50), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 255), 1, 5);
    cv::putText(traj, text2, cv::Point2f(10, 70), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 0, 0), 1, 5);
    cv::imshow( "Trajectory", traj);
    cv::waitKey(0);
    //////////////////////////////////////////////////////////////

//    ///////////////////// Visualize 2 ////////////////////////////////
//    vector<cv::KeyPoint> v_keypoints_map_proj, v_keypoints_2d;
//    for (int i = 0; i < new_frame->features_.size(); i++)
//    {
//        auto current_feature = new_frame->features_[i];
//        if (current_feature->map_point_.lock()) {
//            cv::Point3f p_world = current_feature->map_point_.lock()->pos_;
//            cv::Point2f p_img_proj = new_frame->camera_->world2pixel(p_world, new_frame->Pose());
//            cv::Point2f p_img = current_feature->position_.pt;
//
//            cv::circle(new_frame->rgb_img_, p_img_proj, 3, cv::Scalar(0, 0, 255), cv::FILLED);
//            cv::circle(new_frame->rgb_img_, p_img, 3, cv::Scalar(0, 255, 0), cv::FILLED);
//        }
//    }
//
//    cv::imshow ( "current", new_frame->rgb_img_ );
//    cv::waitKey(0);
//    //////////////////////////////////////////////////////////////

    return success;
}

void VisualOdometry::WritePoseToFile(const string filename)
{
    std::ofstream fout;
    fout.open(filename);
    if (!fout.is_open())
    {
        cout << "my WARNING: failed to store camera trajectory to the wrong file name of:" << endl;
        cout << "    " << filename << endl;
        return;
    }
    for (auto T : cam_pose_history_)
    {
        fout << T.at<double>(0, 0) << " "
             << T.at<double>(0, 1) << " "
             << T.at<double>(0, 2) << " "
             << T.at<double>(0, 3) << " "
             << T.at<double>(1, 0) << " "
             << T.at<double>(1, 1) << " "
             << T.at<double>(1, 2) << " "
             << T.at<double>(1, 3) << " "
             << T.at<double>(2, 0) << " "
             << T.at<double>(2, 1) << " "
             << T.at<double>(2, 2) << " "
             << T.at<double>(2, 3) << "\n";
    }
    fout.close();
}
}


