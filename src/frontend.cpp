//
// Created by jihaorui on 5/2/21.
//

#include <opencv2/opencv.hpp>

#include "myslam/config.h"
#include "myslam/feature.h"
#include "myslam/frontend.h"
#include "myslam/g2o_types.h"
#include "myslam/map.h"
#include "myslam/utils.h"

namespace myslam {

Frontend::Frontend() {

    static const int num_keypoints = Config::Get<int>("number_of_features");
    static const float scale_factor = Config::Get<float>("scale_factor");
    static const int level_pyramid = Config::Get<int>("level_pyramid");
    static const int score_threshold = Config::Get<int>("score_threshold");

    // -- Create ORB
    orb_ = cv::ORB::create(num_keypoints, scale_factor, level_pyramid,
                           31, 0, 2, cv::ORB::HARRIS_SCORE, 31, score_threshold);
}

bool Frontend::AddFrame(Frame::Ptr frame) {
    current_frame_ = frame;
    int num_features = DetectFeatures();

    switch (status_) {
        case FrontendStatus::BLANK:
            current_frame_->SetPose(cv::Mat::eye(4, 4, CV_64F));
            reference_frame_ = current_frame_;
            status_ = FrontendStatus::INITING;
            break;
        case FrontendStatus::INITING:
            MonoInit();
            break;
        case FrontendStatus::TRACKING_GOOD:
        case FrontendStatus::TRACKING_BAD:
            Track();
            break;
        case FrontendStatus::LOST:
            Reset();
            break;
    }

    last_frame_ = current_frame_;
    return true;
}

bool Frontend::MonoInit() {
    MatchFeatures(
            reference_frame_->features_, current_frame_->features_, current_frame_->matches_with_ref_frame_);
    printf("Number of matches with the 1st reference frame: %d\n",
            (int)current_frame_->matches_with_ref_frame_.size());

    EstimateMotionByEpipolarGeometry();

    // Check initialization condition:
    printf("\nCheck VO init conditions: \n");
    // These criteria still needs to be further adjusted
    if (IsGoodToInit_()) {
        bool build_map_success = BuildInitMap();
        if (build_map_success) {
            status_ = FrontendStatus::TRACKING_GOOD;
            return true;
        }
    }
    status_ = FrontendStatus::BLANK;
    return false;
}

bool Frontend::BuildInitMap()
{
    std::vector<cv::Mat> poses{reference_frame_->Pose(), current_frame_->Pose()};
    int cnt_init_landmarks = 0;
    for (int i = 0; i < current_frame_->features_.size(); ++i) {
        // create map point from triangulation
        vector<cv::Point3f> points{
                camera_->pixel2camera(
                        cv::Point2f(reference_frame_->features_[i]->position_.pt.x,
                                reference_frame_->features_[i]->position_.pt.y)),
                camera_->pixel2camera(
                        cv::Point2f(current_frame_->features_[i]->position_.pt.x,
                                    current_frame_->features_[i]->position_.pt.y))};
        cv::Point3f pworld = cv::Point3f(0, 0, 0);

        if (triangulation(poses, points, pworld) && pworld.z > 0) {
            auto new_map_point = MapPoint::CreateNewMappoint();
            new_map_point->SetPos(pworld);
            new_map_point->AddObservation(reference_frame_->features_[i]);
            new_map_point->AddObservation(current_frame_->features_[i]);
            reference_frame_->features_[i]->map_point_= new_map_point;
            current_frame_->features_[i]->map_point_ = new_map_point;
            cnt_init_landmarks++;
            map_->InsertMapPoint(new_map_point);
        }
    }
    current_frame_->SetKeyFrame();
    map_->InsertKeyFrame(current_frame_);
    reference_frame_ = current_frame_;
    backend_->UpdateMap();

    LOG(INFO) << "Initial map created with " << cnt_init_landmarks
              << " map points";

    return true;
}

bool Frontend::Track()
{
    current_frame_->SetPose(relative_motion_ * last_frame_->Pose());

    int num_track_ref_ = TrackRefFrame();
    tracking_inliers_ = EstimateCurrentPoseByPNP();

    if (tracking_inliers_ > num_features_tracking_) {
        // tracking good
        status_ = FrontendStatus::TRACKING_GOOD;
    } else if (tracking_inliers_ > num_features_tracking_bad_) {
        // tracking bad
        status_ = FrontendStatus::TRACKING_BAD;
    } else {
        // lost
        status_ = FrontendStatus::LOST;
    }

    InsertKeyframe();
    relative_motion_ = current_frame_->Pose() * last_frame_->Pose().inv();

    return true;
}

bool Frontend::InsertKeyframe()
{

    if (!CheckConditionForAddKeyFrame()) {
        return false;
    }

    // current frame is a new keyframe
    current_frame_->SetKeyFrame();
    map_->InsertKeyFrame(current_frame_);
    reference_frame_ = current_frame_;

    LOG(INFO) << "Set frame " << current_frame_->id_ << " as keyframe "
              << current_frame_->keyframe_id_;

    SetObservationsForKeyFrame();

    // triangulate map points
    TriangulateNewPoints();

    // update backend because we have a new keyframe
    backend_->UpdateMap();

    return true;
}

bool Frontend::CheckConditionForAddKeyFrame()
{
    if (tracking_inliers_ >= num_features_needed_for_keyframe_) {
        // still have enough features, don't insert keyframe
        return false;
    }
    cv::Mat T_c_r = current_frame_->Pose() * reference_frame_->Pose().inv();
    cv::Mat R, t, rvec;
    getRtFromT(T_c_r, R, t);
    cv::Rodrigues(R, rvec);

    static const auto min_dist_between_two_keyframes = Config::Get<double>("min_dist_between_two_keyframes");
    static const auto min_rotation_angle_betwen_two_keyframes = Config::Get<double>("min_rotation_angle_betwen_two_keyframes");

    double moved_dist = cv::norm(t);
    double rotated_angle = cv::norm(rvec);

    printf("Wrt prev keyframe, relative dist = %.5f, angle = %.5f\n", moved_dist, rotated_angle);

    // Satisfy each one will be a good keyframe
    return moved_dist > min_dist_between_two_keyframes ||
           rotated_angle > min_rotation_angle_betwen_two_keyframes;
}

void Frontend::SetObservationsForKeyFrame()
{
    for (auto &feat : current_frame_->features_) {
        auto mp = feat->map_point_.lock();
        if (mp) mp->AddObservation(feat);
    }
}

int Frontend::TriangulateNewPoints() {
    cv::Mat current_pose_Twc = current_frame_->Pose().inv();
    int cnt_triangulated_pts = 0;
    for (int i = 0; i < current_frame_->features_.size(); ++i) {
        if (current_frame_->features_[i]->map_point_.expired()) {
            // 如果当前帧的特征点没有关联地图点,则利用当前位姿将特征点反投影到地图上构建新的地图点
            cv::Point3f p_cam =
                    camera_->pixel2camera(
                            cv::Point2f(current_frame_->features_[i]->position_.pt.x,
                                        current_frame_->features_[i]->position_.pt.y));

            auto new_map_point = MapPoint::CreateNewMappoint();
            cv::Mat pworld_mat = current_pose_Twc * point3f_to_mat3x1(p_cam);
            cv::Point3f pworld = cv::Point3f(pworld_mat.at<double>(0, 0),
                                 pworld_mat.at<double>(1, 0),
                                 pworld_mat.at<double>(2, 0));
            new_map_point->SetPos(pworld);
            new_map_point->AddObservation(
                    current_frame_->features_[i]);

            current_frame_->features_[i]->map_point_ = new_map_point;
            map_->InsertMapPoint(new_map_point);
            cnt_triangulated_pts++;
        }
    }
    LOG(INFO) << "new landmarks: " << cnt_triangulated_pts;
    return cnt_triangulated_pts;
}

int Frontend::EstimateMotionByEpipolarGeometry()
{
    vector<cv::Point2f> pts_in_img1;
    vector<cv::Point2f> pts_in_img2;
    for (int i = 0; i < current_frame_->matches_with_ref_frame_.size(); i++) {
        cv::DMatch match = current_frame_->matches_with_ref_frame_[i];
        pts_in_img1.push_back(reference_frame_->features_[match.queryIdx]->position_.pt);
        pts_in_img2.push_back(current_frame_->features_[match.queryIdx]->position_.pt);
    }

    // -- Essential matrix
    static auto findEssentialMat_prob = Config::Get<double>("findEssentialMat_prob");
    static auto findEssentialMat_threshold = Config::Get<double>("findEssentialMat_threshold");
    cv::Mat inliers_mask;
    cv::Mat essential_matrix;
    essential_matrix = findEssentialMat(
            pts_in_img1, pts_in_img2, camera_->K_,
            cv::RANSAC, findEssentialMat_prob, findEssentialMat_threshold, inliers_mask);
    essential_matrix /= essential_matrix.at<double>(2, 2);

    // Get inliers
    vector<int> inliers_index;
    for (int i = 0; i < inliers_mask.rows; i++)
    {
        if ((int)inliers_mask.at<unsigned char>(i, 0) == 1)
        {
            inliers_index.push_back(i);
        }
    }

    // Recover R,t from Essential matrix
    cv::Mat R, t;
    recoverPose(essential_matrix, pts_in_img1, pts_in_img2, camera_->K_, R, t, inliers_mask);

    // Normalize t
    t = t / cv::norm(t);

    current_frame_->SetPose(convertRt2T(R, t));
}

int Frontend::EstimateCurrentPoseByPNP()
{
    vector<cv::Point3f> pts_3d;
    vector<cv::Point2f> pts_2d;
    for (int i = 0; i < current_frame_->features_.size(); i++) {
        auto mappoint = current_frame_->features_[i]->map_point_.lock();
        if (!mappoint) {
            pts_3d.push_back(mappoint->pos_);
            pts_2d.push_back(current_frame_->features_[i]->position_.pt);
        }
    }
    int num_matches = pts_3d.size();
    cout << "Number of 3d-2d pairs: " << num_matches << endl;

    // -- Solve PnP, get T_c_w_
    constexpr int kMinPtsForPnP = 5;

    cv::Mat pnp_inliers_mask; // type = 32SC1, size = 999x1
    cv::Mat rvec, t;

    bool is_pnp_good = num_matches >= kMinPtsForPnP;
    int num_inliers = 0;
    if (is_pnp_good) {
        bool useExtrinsicGuess = false;
        int iterationsCount = 100;
        float reprojectionError = 2.0;
        double confidence = 0.999;
        cv::solvePnPRansac(pts_3d, pts_2d, camera_->K_, cv::Mat(), rvec, t,
                           useExtrinsicGuess,
                           iterationsCount, reprojectionError, confidence, pnp_inliers_mask, cv::SOLVEPNP_EPNP);
        int num_inliers = pnp_inliers_mask.rows;

        cv::Mat R;
        cv::Rodrigues(rvec, R); // angle-axis rotation to 3x3 rotation matrix

        // -- Update current camera pos
        current_frame_->SetPose(convertRt2T(R, t));
    }

    return num_inliers;
}

//int Frontend::EstimateCurrentPoseByG2O() {
//    // setup g2o
//    typedef g2o::BlockSolver_6_3 BlockSolverType;
//    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>
//            LinearSolverType;
//    auto solver = new g2o::OptimizationAlgorithmLevenberg(
//            g2o::make_unique<BlockSolverType>(
//                    g2o::make_unique<LinearSolverType>()));
//    g2o::SparseOptimizer optimizer;
//    optimizer.setAlgorithm(solver);
//
//    // vertex
//    VertexPose *vertex_pose = new VertexPose();  // camera vertex_pose
//    vertex_pose->setId(0);
//    vertex_pose->setEstimate(current_frame_->Pose());
//    optimizer.addVertex(vertex_pose);
//
//    // K
//    Mat33 K = camera_left_->K();
//
//    // edges
//    int index = 1;
//    std::vector<EdgeProjectionPoseOnly *> edges;
//    std::vector<Feature::Ptr> features;
//    for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
//        auto mp = current_frame_->features_left_[i]->map_point_.lock();
//        if (mp) {
//            features.push_back(current_frame_->features_left_[i]);
//            EdgeProjectionPoseOnly *edge =
//                    new EdgeProjectionPoseOnly(mp->pos_, K);
//            edge->setId(index);
//            edge->setVertex(0, vertex_pose);
//            edge->setMeasurement(
//                    toVec2(current_frame_->features_left_[i]->position_.pt));
//            edge->setInformation(Eigen::Matrix2d::Identity());
//            edge->setRobustKernel(new g2o::RobustKernelHuber);
//            edges.push_back(edge);
//            optimizer.addEdge(edge);
//            index++;
//        }
//    }
//
//    // estimate the Pose the determine the outliers
//    const double chi2_th = 5.991;
//    int cnt_outlier = 0;
//    for (int iteration = 0; iteration < 4; ++iteration) {
//        vertex_pose->setEstimate(current_frame_->Pose());
//        optimizer.initializeOptimization();
//        optimizer.optimize(10);
//        cnt_outlier = 0;
//
//        // count the outliers
//        for (size_t i = 0; i < edges.size(); ++i) {
//            auto e = edges[i];
//            if (features[i]->is_outlier_) {
//                e->computeError();
//            }
//            if (e->chi2() > chi2_th) {
//                features[i]->is_outlier_ = true;
//                e->setLevel(1);
//                cnt_outlier++;
//            } else {
//                features[i]->is_outlier_ = false;
//                e->setLevel(0);
//            };
//
//            if (iteration == 2) {
//                e->setRobustKernel(nullptr);
//            }
//        }
//    }
//
//    LOG(INFO) << "Outlier/Inlier in pose estimating: " << cnt_outlier << "/"
//              << features.size() - cnt_outlier;
//    // Set pose and outlier
//    current_frame_->SetPose(vertex_pose->estimate());
//
//    LOG(INFO) << "Current Pose = \n" << current_frame_->Pose().matrix();
//
//    for (auto &feat : features) {
//        if (feat->is_outlier_) {
//            feat->map_point_.reset();
//            feat->is_outlier_ = false;  // maybe we can still use it in future
//        }
//    }
//    return features.size() - cnt_outlier;
//}

int Frontend::TrackRefFrame()
{
    // use LK flow to estimate points in the right image

    MatchFeatures(
            reference_frame_->features_, current_frame_->features_, current_frame_->matches_with_ref_frame_);

    int num_good_pts = 0;

    for (int i = 0; i < current_frame_->matches_with_ref_frame_.size(); ++i) {
        cv::DMatch match = current_frame_->matches_with_ref_frame_[i];
        auto feature = current_frame_->features_[match.trainIdx];
        feature->map_point_ = reference_frame_->features_[match.queryIdx]->map_point_;
        num_good_pts++;
    }

    LOG(INFO) << "Find " << num_good_pts << " in the last image.";
    return num_good_pts;
}


bool Frontend::Reset()
{
    LOG(INFO) << "Reset whole system. ";
    current_frame_->SetPose(relative_motion_ * last_frame_->Pose());
    reference_frame_ = current_frame_;
    status_ = FrontendStatus::INITING;
    return true;
}

int Frontend::DetectFeatures() {
    vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    // detect keypoints and descriptors
    orb_->detect(current_frame_->rgb_img_, keypoints);
    orb_->compute(current_frame_->rgb_img_, keypoints, descriptors);

    int cnt_detected = 0;
    for (int i = 0; i < keypoints.size(); i++)
    {
        current_frame_->features_.push_back(
                Feature::Ptr(new Feature(current_frame_, keypoints[i], descriptors.row(i))));
        cnt_detected++;
    }

    LOG(INFO) << "Detect " << cnt_detected << " new features";
    return cnt_detected;
}


}  // namespace myslam

