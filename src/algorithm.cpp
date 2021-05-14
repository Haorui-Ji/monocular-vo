//
// Created by jihaorui on 5/7/21.
//

#include "myslam/algorithm.h"
#include "myslam/config.h"
#include "myslam/utils.h"

namespace myslam
{
std::pair<vector<int>, vector<double>> FindBestMatches(
        const cv::Mat &descriptors_1,
        const cv::Mat &descriptors_2,
        vector<cv::DMatch> &matches)
{
    // find best matches
    vector<int> idx(descriptors_1.rows);
    vector<double> dist(descriptors_1.rows);
    for (int i = 0; i < descriptors_1.rows; i++) {
        int bestMatch = -1;
        double bestDist = 1e10;
        for (int j = 0; j < matches.size(); j++) {
            if (matches[j].queryIdx == i && matches[j].distance < bestDist) {
                bestDist = matches[j].distance;
                bestMatch = matches[j].trainIdx;
            }
        }
        idx[i] = bestMatch;
        dist[i] = bestDist;
    }

    // We still need to consider the possibility that for a feature in feature_1,
    // there's no feature in feature_2 that corresponds with it
    return std::make_pair(idx, dist);
}

std::pair<vector<int>, vector<double>> MatchFeaturesHelper(
        const cv::Mat &descriptors_1,
        const cv::Mat &descriptors_2,
        vector<cv::DMatch> &matches)
{
    double match_ratio = Config::Get<double>("match_ratio");
    cv::FlannBasedMatcher matcher_flann(new cv::flann::LshIndexParams(5, 10, 2));
    vector<cv::DMatch> all_matches;

    matcher_flann.match(descriptors_1, descriptors_2, all_matches);

    double min_dist = 10000, max_dist = 0;

    for (int i = 0; i < all_matches.size(); i++) {
        double dist = all_matches[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    // Select good matches and push to the result vector.
    for (cv::DMatch &m : all_matches) {
        if (m.distance <= max(min_dist * match_ratio, 50.0)) {
            matches.push_back(m);
        }
    }

    // find best matches
    return FindBestMatches(descriptors_1, descriptors_2, matches);
}

void MatchFeatures(
        const vector<std::shared_ptr<Feature>> &features_1,
        const vector<std::shared_ptr<Feature>> &features_2,
        vector<cv::DMatch> &matches)
{
    int rows_1 = features_1.size();
    int cols_1 = features_1[0]->descriptor_.cols;

    int rows_2 = features_2.size();
    int cols_2 = features_2[0]->descriptor_.cols;

    cv::Mat descriptors_1(rows_1, cols_1, CV_8U);
    cv::Mat descriptors_2(rows_2, cols_2, CV_8U);

    for (int i = 0; i < features_1.size(); i++ )
    {
        features_1[i]->descriptor_.copyTo(descriptors_1.row(i));
    }
    for (int j = 0; j < features_2.size(); j++ )
    {
        features_2[j]->descriptor_.copyTo(descriptors_2.row(j));
    }

    vector<cv::DMatch> all_matches_12, all_matches_21;

    std::pair<vector<int>, vector<double>> best_matches_12 = MatchFeaturesHelper(descriptors_1, descriptors_2,
                                                                                 all_matches_12);
    std::pair<vector<int>, vector<double>> best_matches_21 = MatchFeaturesHelper(descriptors_2, descriptors_1,
                                                                                 all_matches_21);

    for (int idx1 = 0; idx1 < (int) best_matches_12.first.size(); idx1++) {
        int idx2 = best_matches_12.first[idx1];
        if (idx2 != -1) // it means we can find corresponding matches in feature_2
        {
            if (best_matches_21.first[idx2] == idx1) {
                cv::DMatch match = cv::DMatch(idx1, idx2, best_matches_12.second[idx1]);
                matches.push_back(match);
            }
        }
    }
}

void Triangulation(
        const vector<cv::Point2f> &inlier_pts_in_img1,
        const vector<cv::Point2f> &inlier_pts_in_img2,
        const Camera::Ptr &camera,
        const cv::Mat &R21, const cv::Mat &t21,
        vector<cv::Point3f> &pts3d_in_cam1)
{
    const cv::Mat &K = camera->K_;

    // back project to camera coordinates on normalized plane
    vector<cv::Point2f> inlier_pts_in_cam1, inlier_pts_in_cam2;
    for (int idx = 0; idx < inlier_pts_in_img1.size(); idx++)
    {
        cv::Point3f tmp1 = camera->pixel2camera(inlier_pts_in_img1[idx], 1);
        cv::Point3f tmp2 = camera->pixel2camera(inlier_pts_in_img2[idx], 1);
        inlier_pts_in_cam1.push_back(cv::Point2f(tmp1.x, tmp1.y));
        inlier_pts_in_cam2.push_back(cv::Point2f(tmp2.x, tmp2.y));
    }

    // set up
    cv::Mat T_c1_w =
            (cv::Mat_<double>(3, 4) <<
                    1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0);
    cv::Mat T_c2_w = convertRt2T_3x4(R21, t21);

    // triangulartion
    cv::Mat pts4d_in_world;
    cv::triangulatePoints(
            T_c1_w, T_c2_w, inlier_pts_in_cam1, inlier_pts_in_cam2, pts4d_in_world);

    // change to homogeneous coords
    vector<cv::Point3f> pts3d_in_world;
    for (int i = 0; i < pts4d_in_world.cols; i++)
    {
        cv::Mat x = pts4d_in_world.col(i);
        x /= x.at<float>(3, 0);
        cv::Point3f pt3d_in_world(
                x.at<float>(0, 0),
                x.at<float>(1, 0),
                x.at<float>(2, 0));
        pts3d_in_world.push_back(pt3d_in_world);
    }

    // return
    pts3d_in_cam1 = pts3d_in_world;
}

vector<bool> CheckGoodTriangulationResult(
        const cv::Mat &pose_1,
        const cv::Mat &pose_2,
        const Camera::Ptr &camera,
        const vector<cv::Point3f> pts3d_in_cam1,
        const vector<cv::Point2f> inlier_pts_in_img_1,
        const vector<cv::Point2f> inlier_pts_in_img_2)
{
    cv::Mat T21 = pose_2 * pose_1.inv();
    vector<cv::Point3f> pts3d_in_cam2;
    for (const cv::Point3f &p1 : pts3d_in_cam1)
    {
        pts3d_in_cam2.push_back(transCoord(p1, T21));
    }

    int N = (int)pts3d_in_cam2.size();

    // Step 1: Remove triangulation results whose depth < 0 or any component is infinite
    vector<bool> feasibility;
    for (int i = 0; i < N; i++)
    {
        cv::Point3f &p_in_cam2 = pts3d_in_cam2[i];
        feasibility.push_back(p_in_cam2.z >= 0 &&
                              isfinite(p_in_cam2.x) &&
                              isfinite(p_in_cam2.y) &&
                              isfinite(p_in_cam2.z));
    }

//    // Step 2: Remove those with a too large or too small parallax angle.
//    static const double min_triang_angle = Config::Get<double>("min_triang_angle");
//    static const double max_ratio_between_max_angle_and_median_angle =
//            Config::Get<double>("max_ratio_between_max_angle_and_median_angle");
//
//    vector<double> angles;
//    // -- Compute parallax angles
//    for (int i = 0; i < N; i++)
//    {
//        cv::Point3f &p_in_cam2 = pts3d_in_cam2[i];
//        cv::Mat p_in_world = point3f_to_mat3x1(pts3d_in_cam1[i]);
//        cv::Mat vec_p_to_cam_curr = getPosFromT(pose_2) - p_in_world;
//        cv::Mat vec_p_to_cam_prev = getPosFromT(pose_1) - p_in_world;
//        double angle = calcAngleBetweenTwoVectors(vec_p_to_cam_curr, vec_p_to_cam_prev);
//        angles.push_back(angle / 3.1415926 * 180.0);
//    }
//
//    // Get statistics
//    vector<double> sort_a = angles;
//    sort(sort_a.begin(), sort_a.end());
//    double mean_angle = accumulate(sort_a.begin(), sort_a.end(), 0.0) / N;
//    double median_angle = sort_a[N / 2];
//    printf("Triangulation angle: mean=%f, median=%f, min=%f, max=%f\n",
//           mean_angle,   // mean
//           median_angle, // median
//           sort_a[0],    // min
//           sort_a[N - 1] // max
//    );
//
//    for (int i = 0; i < N; i++)
//    {
//        if (angles[i] < min_triang_angle ||
//            angles[i] / median_angle > max_ratio_between_max_angle_and_median_angle)
//        {
//            feasibility[i] = false;
//        }
//    }
////    ///////////////////////////////////
////    // Test
////    int cnt_false_2 = 0;
////    int cnt_true_2 = 0;
////    for (int i = 0; i < N; i++)
////    {
////        if (feasibility[i] == false)
////        {
////            cnt_false_2++;
////        }
////        else
////        {
////            cnt_true_2++;
////        }
////    }
////    printf("Stage 2:\n Total: %d, true: %d, false: %d\n", N, cnt_true_2, cnt_false_2);
////    ///////////////////////////////////
//
//
    // Step 3: Remove those reprojection error is too large
    cv::Mat &K = camera->K_;
    static const double sigma = Config::Get<double>("initialization_sigma");
    double sigma2 = sigma * sigma;
    for (int i = 0; i < N; i++)
    {
        cv::Point3f p_cam1 = pts3d_in_cam1[i];
        cv::Point2f p_img1_proj = camera->camera2pixel(p_cam1);

        cv::Point3f p_cam2 = pts3d_in_cam2[i];
        cv::Point2f p_img2_proj = camera->camera2pixel(p_cam2);

        // Check frame1
        cv::Point2f pt1 = inlier_pts_in_img_1[i];
        float squareError1 = (p_img1_proj.x - pt1.x) * (p_img1_proj.x - pt1.x) \
                            + (p_img1_proj.y - pt1.y) * (p_img1_proj.y - pt1.y);

        // Check frame 2
        cv::Point2f pt2 = inlier_pts_in_img_2[i];
        float squareError2 = (p_img2_proj.x - pt2.x) * (p_img2_proj.x - pt2.x) \
                            + (p_img2_proj.y - pt2.y) * (p_img2_proj.y - pt2.y);

        if (squareError1 > 4*sigma2 || squareError2 > 4*sigma2)
        {
            feasibility[i] = false;
        }
    }

//    ///////////////////////////////////
//    // Test
//    int cnt_false_3 = 0;
//    int cnt_true_3 = 0;
//    for (int i = 0; i < N; i++)
//    {
//        if (feasibility[i] == false)
//        {
//            cnt_false_3++;
//        }
//        else
//        {
//            cnt_true_3++;
//        }
//    }
//    printf("Stage 3:\n Total: %d, true: %d, false: %d\n", N, cnt_true_3, cnt_false_3);
//    ///////////////////////////////////

    return feasibility;

}

}
