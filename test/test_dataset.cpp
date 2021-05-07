//
// Created by jihaorui on 5/2/21.
//

#include "myslam/common_include.h"
#include "myslam/config.h"
#include "myslam/dataset.h"

using namespace myslam;

int main(int argc, char** argv) {

    if (argc < 2) {
        cout << "Usage: config_file sequence_number. e.g. ./config/kitti.yaml 00" << endl;
        return 1;
    }

    if (!Config::SetParameterFile(argv[1])) {
        return false;
    }

    string dataset_root_path = Config::Get<string>("dataset_dir") + argv[2];
    Dataset::Ptr dataset = Dataset::Ptr(new Dataset(dataset_root_path));

    CHECK_EQ(dataset->Init(), true);

    vector<double> times = dataset->GetTimeStamp();
    cout << times.size() << endl;
    for (int i = 0; i < times.size(); i++){
        cout << times[i] << endl;
    }


    cout << dataset->GetCamera(2)->K_ << endl;
    cout << "Kitti Dataset Test Complete" << endl;
}


