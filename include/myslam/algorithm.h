//
// Created by jihaorui on 5/7/21.
//

// algorithms used in myslam
#include "myslam/common_include.h"
#include "myslam/feature.h"

namespace myslam {

/**
 * linear triangulation with SVD
 * @param poses     poses,
 * @param points    points in normalized plane
 * @param pt_world  triangulated point in the world
 * @return true if success
 */
inline bool triangulation(const std::vector<cv::Mat> &poses,
                          const std::vector<Vec3> points, Vec3 &pt_world)
{
    MatXX A(2 * poses.size(), 4);
    VecX b(2 * poses.size());
    b.setZero();
    for (size_t i = 0; i < poses.size(); ++i) {
        Mat34 m = poses[i].matrix3x4();
        A.block<1, 4>(2 * i, 0) = points[i][0] * m.row(2) - m.row(0);
        A.block<1, 4>(2 * i + 1, 0) = points[i][1] * m.row(2) - m.row(1);
    }
    auto svd = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
    pt_world = (svd.matrixV().col(3) / svd.matrixV()(3, 3)).head<3>();

    if (svd.singularValues()[3] / svd.singularValues()[2] < 1e-2) {
        // 解质量不好，放弃
        return true;
    }
    return false;
}


void MatchFeatures(
        vector<std::shared_ptr<Feature>> features_1,
        const cv::Mat &descriptors_2,
        vector<cv::DMatch> &matches);

// --------------------- Assistant functions ---------------------
std::pair<vector<int>, vector<double>> FindBestMatches(
        const cv::Mat &descriptors_1,
        const cv::Mat &descriptors_2,
        vector<cv::DMatch> &matches);

std::pair<vector<int>, vector<double>> MatchFeaturesHelper(
        const cv::Mat &descriptors_1,
        const cv::Mat &descriptors_2,
        vector<cv::DMatch> &matches);

}  // namespace myslam

#endif  // MYSLAM_ALGORITHM_H
