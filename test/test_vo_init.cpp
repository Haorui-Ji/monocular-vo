//
// Created by jihaorui on 5/8/21.
//

#include "myslam/common_include.h"
#include "myslam/config.h"
#include "myslam/dataset.h"
#include "myslam/visual_odometry.h"
#include "myslam/utils.h"

using namespace myslam;

int main(int argc, char** argv) {

    if (argc < 1) {
        cout << "Usage: config_file. e.g. ./config/kitti.yaml" << endl;
        return 1;
    }

    string config_file_path = argv[1];
    VisualOdometry::Ptr vo(new VisualOdometry(config_file_path));
    assert(vo->Init() == true);

    vo->Run();

    // Save camera trajectory
    const string save_predicted_traj_to = Config::Get<string>("save_predicted_traj_to");
    vo->WritePoseToFile(save_predicted_traj_to);

//    /////////////////// Visualize whole trajectory ///////////////////////////////
//    cv::Mat traj = Mat::zeros(600, 600, CV_8UC3);
//    vector<cv::Mat> cam_pose_history = vo->GetCamPoseHistory();
//    cout << cam_pose_history.size();
//
//    for (int i = 0; i < cam_pose_history.size(); i++) {
//        cv::Mat R, t_vec;
//        getRtFromT(cam_pose_history[i], R, t_vec);
//        vector<vector<float>> poses = vo->GetDataset()->GetGroundTruthPose();
//        string text  = "Red color: estimated trajectory";
//        string text2 = "Blue color: Groundtruth trajectory";
//
//        t_vec.convertTo(t_vec, CV_32F);
//        cv::Point2f center = cv::Point2f(int(t_vec.at<float>(0)) + 300, int(t_vec.at<float>(2)) + 100);
//        cv::Point2f t_center = cv::Point2f(int(poses[i][3]) + 300, int(poses[i][11]) + 100);
//        cv::circle(traj, center, 1, cv::Scalar(0, 0, 255), 2);
//        cv::circle(traj, t_center, 1, cv::Scalar(255, 0, 0), 2);
//        cv::rectangle(traj, cv::Point2f(10, 30), cv::Point2f(550, 50),  cv::Scalar(0,0,0), cv::FILLED);
//        cv::putText(traj, text, cv::Point2f(10, 50), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 255), 1, 5);
//        cv::putText(traj, text2, cv::Point2f(10, 70), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 0, 0), 1, 5);
//        cv::imshow( "Trajectory", traj);
//        cv::waitKey(0);
//    }
//
//    //////////////////////////////////////////////////////////////

}