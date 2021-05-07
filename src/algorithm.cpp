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
    vector<int> idx(descriptors_1.rows, -1);
    vector<double> dist(descriptors_1.rows, -1);
    for (int i = 0; i < descriptors_1.rows; i++) {
        double bestDist = 1e10;
        int bestMatch = -1;
        for (int j = 0; j < matches.size(); j++) {
            if (matches[j].queryIdx == i && matches[j].distance < bestDist) {
                bestDist = matches[j].distance;
                bestMatch = matches[j].trainIdx;
            }
        }
        idx[i] = bestMatch;
        dist[i] = bestDist;
    }

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
        const cv::Mat &descriptors_1,
        const cv::Mat &descriptors_2,
        vector<cv::DMatch> &matches)
{
    vector<cv::DMatch> all_matches_12, all_matches_21;

    std::pair<vector<int>, vector<double>> best_matches_12 = MatchFeaturesHelper(descriptors_1, descriptors_2,
                                                                                 all_matches_12);
    std::pair<vector<int>, vector<double>> best_matches_21 = MatchFeaturesHelper(descriptors_2, descriptors_1,
                                                                                 all_matches_21);

    for (int idx1 = 0; idx1 < (int) best_matches_12.first.size(); idx1++) {
        int idx2 = best_matches_12.first[idx1];
        if (best_matches_21.first[idx2] == idx1) {
            cv::DMatch match = cv::DMatch(idx1, idx2, best_matches_12.second[idx1]);
            matches.push_back(match);
        }
    }
}

}