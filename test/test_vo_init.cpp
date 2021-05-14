//
// Created by jihaorui on 5/8/21.
//

#include "myslam/common_include.h"
#include "myslam/config.h"
#include "myslam/dataset.h"
#include "myslam/visual_odometry.h"

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

}