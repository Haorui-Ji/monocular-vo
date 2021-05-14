//
// Created by jihaorui on 5/2/21.
//

#include <opencv2/opencv.hpp>

#include "myslam/config.h"
#include "myslam/algorithm.h"
#include "myslam/feature.h"
#include "myslam/frontend.h"
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

    DetectFeatures();

    LOG(INFO) << "Current status: " << static_cast<std::underlying_type<FrontendStatus>::type>(status_);

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

    if (last_frame_ != nullptr) {
        cv::Mat relative_motion = current_frame_->Pose() * last_frame_->Pose().inv();
        cv::Mat t = getPosFromT(current_frame_->Pose());
        cv::Mat t_relative = getPosFromT(relative_motion);
        cout << "\nCamera motion:" << endl;
        cout << "Motion w.r.t World:\n " << t << endl;
        cout << "Motion w.r.t last frame:\n " << t_relative << endl;
    }

    last_frame_ = current_frame_;
//
//    cout << map_->GetAllMapPoints().size() << endl;
//    for (auto &mp: map_->GetAllMapPoints()) {
//        cout << mp.second->pos_ << endl;
//    }

    return true;
}

bool Frontend::MonoInit() {

    MatchFeatures(
            reference_frame_->features_, current_frame_->features_, current_frame_->matches_with_ref_frame_);
    LOG(INFO) << "Number of matches with the 1st reference frame:" << current_frame_->matches_with_ref_frame_.size();

    tracking_inliers_ = EstimateMotionByEpipolarGeometry();

    // Check initialization condition:
    printf("\nCheck VO init conditions: \n");
    // These criteria still needs to be further adjusted
    if (IsGoodToInit()) {
        bool build_map_success = BuildInitMap();
        if (build_map_success) {
            status_ = FrontendStatus::TRACKING_GOOD;
            return true;
        }
    }
    status_ = FrontendStatus::BLANK;
    return false;
}

bool Frontend::IsGoodToInit()
{
    static const int min_inlier_matches = Config::Get<int>("min_inlier_matches");
    static const double min_pixel_dist = Config::Get<double>("min_pixel_dist");
    static const double min_median_triangulation_angle = Config::Get<double>("min_median_triangulation_angle");

    // -- Check CRITERIA_0: num inliers should be large
    bool criteria_0 = true;
    if (tracking_inliers_ < min_inlier_matches)
    {
        printf("%d inlier points are too few... threshold is %d.\n",
               tracking_inliers_, min_inlier_matches);
        criteria_0 = false;
    }

    bool criteria = criteria_0;

    return criteria;
}

bool Frontend::BuildInitMap()
{
    cv::Mat R, t;
    getRtFromT(relative_motion_, R, t);

    int cnt_init_landmarks = 0;
    vector<std::shared_ptr<Feature>> inlier_features_ref, inlier_features_curr;
    vector<cv::Point2f> inlier_pts_in_ref_frame, inlier_pts_in_curr_frame;

    // Extract inlier matches between frames
    for (int i = 0; i < current_frame_->matches_with_ref_frame_.size(); i++)
    {
        cv::DMatch match = current_frame_->matches_with_ref_frame_[i];
        if (current_frame_->features_[match.trainIdx]->is_inlier_)
        {
            inlier_features_ref.push_back(reference_frame_->features_[match.queryIdx]);
            inlier_features_curr.push_back(current_frame_->features_[match.trainIdx]);
            inlier_pts_in_ref_frame.push_back(reference_frame_->features_[match.queryIdx]->position_.pt);
            inlier_pts_in_curr_frame.push_back(current_frame_->features_[match.trainIdx]->position_.pt);
        }

    }

    vector<cv::Point3f> p_world;
    Triangulation(inlier_pts_in_ref_frame, inlier_pts_in_curr_frame, camera_, R, t, p_world);

    vector<bool> feasibility = CheckGoodTriangulationResult(
            reference_frame_->Pose(), current_frame_->Pose(), camera_,
            p_world, inlier_pts_in_ref_frame, inlier_pts_in_curr_frame);

    for (int i = 0; i < feasibility.size(); i++)
    {
        if (feasibility[i])
        {
            auto new_map_point = MapPoint::CreateNewMappoint();
            new_map_point->SetPos(p_world[i]);
            new_map_point->AddObservation(inlier_features_ref[i]);
            new_map_point->AddObservation(inlier_features_curr[i]);
            inlier_features_ref[i]->map_point_= new_map_point;
            inlier_features_curr[i]->map_point_ = new_map_point;
            inlier_features_curr[i]->associate_new_map_point_ = true;
            cnt_init_landmarks++;
            map_->InsertMapPoint(new_map_point);
        }
    }

    current_frame_->SetKeyFrame();
    map_->InsertKeyFrame(current_frame_);
    reference_frame_ = current_frame_;
//    backend_->UpdateMap();


    LOG(INFO) << "Initial map created with " << cnt_init_landmarks
              << " map points";

    return true;
}

bool Frontend::Track()
{
    current_frame_->SetPose(relative_motion_ * last_frame_->Pose());

    TrackLastFrame();
    tracking_inliers_ = EstimateCurrentPoseByPNP();

    LOG(INFO) << "Tracking inliers: " << tracking_inliers_;

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

    if (CheckConditionForAddKeyFrame() == false) {
        return false;
    }

    // current frame is a new keyframe
    current_frame_->SetKeyFrame();
    map_->InsertKeyFrame(current_frame_);

    LOG(INFO) << "Set frame " << current_frame_->id_ << " as keyframe "
              << current_frame_->keyframe_id_;

    SetObservationsForKeyFrame();

    // triangulate map points
    TriangulateNewPoints();

    reference_frame_ = current_frame_;

    // update backend because we have a new keyframe
//    backend_->UpdateMap();

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

int Frontend::TriangulateNewPoints()
{
    /// 这一步有问题,有些 feature 没有 descriptor
    MatchFeatures(
            reference_frame_->features_, current_frame_->features_, current_frame_->matches_with_ref_frame_);
    LOG(INFO) << "Number of matches with the previous reference frame:" << current_frame_->matches_with_ref_frame_.size();

    cv::Mat R, t;
    cv::Mat relative_motion_with_ref = current_frame_->Pose() * reference_frame_->Pose().inv();
    getRtFromT(relative_motion_with_ref, R, t);

    int cnt_triangulated_pts = 0;
    vector<std::shared_ptr<Feature>> inlier_features_ref, inlier_features_curr;
    vector<cv::Point2f> inlier_pts_in_ref_frame, inlier_pts_in_curr_frame;

    // Extract inlier matches between frames
    for (int i = 0; i < current_frame_->matches_with_ref_frame_.size(); i++)
    {
        cv::DMatch match = current_frame_->matches_with_ref_frame_[i];
        if (reference_frame_->features_[match.queryIdx]->map_point_.expired() &&
                current_frame_->features_[match.trainIdx]->map_point_.expired())
        {
            // 如果当前帧和参考帧匹配上的特征点均未关联地图点,则进行三角化
            inlier_features_ref.push_back(reference_frame_->features_[match.queryIdx]);
            inlier_features_curr.push_back(current_frame_->features_[match.trainIdx]);
            inlier_pts_in_ref_frame.push_back(reference_frame_->features_[match.queryIdx]->position_.pt);
            inlier_pts_in_curr_frame.push_back(current_frame_->features_[match.trainIdx]->position_.pt);
        }

    }

    vector<cv::Point3f> p_3d_ref, p_world;
    Triangulation(inlier_pts_in_ref_frame, inlier_pts_in_curr_frame, camera_, R, t, p_3d_ref);

    vector<bool> feasibility = CheckGoodTriangulationResult(
            reference_frame_->Pose(), current_frame_->Pose(), camera_,
            p_3d_ref, inlier_pts_in_ref_frame, inlier_pts_in_curr_frame);

    for (int i = 0; i < p_3d_ref.size(); i++) {
        p_world.push_back(transCoord(p_3d_ref[i], reference_frame_->Pose().inv()));
    }

    for (int i = 0; i < feasibility.size(); i++)
    {
        if (feasibility[i])
        {
            auto new_map_point = MapPoint::CreateNewMappoint();
            new_map_point->SetPos(p_world[i]);
            new_map_point->AddObservation(inlier_features_ref[i]);
            new_map_point->AddObservation(inlier_features_curr[i]);
            inlier_features_ref[i]->map_point_= new_map_point;
            inlier_features_curr[i]->map_point_ = new_map_point;
            inlier_features_curr[i]->associate_new_map_point_ = true;
            cnt_triangulated_pts++;
            map_->InsertMapPoint(new_map_point);
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
        pts_in_img2.push_back(current_frame_->features_[match.trainIdx]->position_.pt);
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
    int num_inliers = 0;
    for (int i = 0; i < inliers_mask.rows; i++)
    {
        if ((int)inliers_mask.at<unsigned char>(i, 0) == 1)
        {
            cv::DMatch match = current_frame_->matches_with_ref_frame_[i];
            current_frame_->features_[match.trainIdx]->is_inlier_ = true;
            num_inliers++;
        }
    }

    // Recover R,t from Essential matrix
    cv::Mat R, t;
    recoverPose(essential_matrix, pts_in_img1, pts_in_img2, camera_->K_, R, t, inliers_mask);

    // Normalize t
    t = t / cv::norm(t);

    relative_motion_ = convertRt2T(R, t);
    current_frame_->SetPose(relative_motion_ * reference_frame_->Pose());

    return num_inliers;
}

int Frontend::EstimateCurrentPoseByPNP()
{
    vector<cv::Point3f> pts_3d;
    vector<cv::Point2f> pts_2d;
    vector<std::shared_ptr<Feature>> features;
    for (int i = 0; i < current_frame_->features_.size(); i++) {
        auto mappoint = current_frame_->features_[i]->map_point_.lock();
        if (mappoint) {
            pts_3d.push_back(mappoint->pos_);
            pts_2d.push_back(current_frame_->features_[i]->position_.pt);
            features.push_back(current_frame_->features_[i]);
        }
    }
    int num_matches = pts_3d.size();
    LOG(INFO) << "Number of 3d-2d pairs: " << num_matches;

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

        num_inliers = pnp_inliers_mask.rows;
        for (int i = 0; i < num_inliers; i++)
        {
            int good_idx = pnp_inliers_mask.at<int>(i, 0);

            features[good_idx]->is_inlier_ = true;
        }

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

int Frontend::TrackLastFrame()
{
    MatchFeatures(
            last_frame_->features_, current_frame_->features_, current_frame_->matches_with_last_frame_);

    int num_good_pts = 0;

    ///////////////////////////////////
    int count = 0;
    for (int i = 0; i < last_frame_->features_.size(); i++) {
        auto last_feature = last_frame_->features_[i];
        if (last_feature->map_point_.lock() && (last_feature->is_inlier_ == true ||
                last_feature->associate_new_map_point_ == true)) {
            count ++;
        }
    }
    LOG(INFO) << "There are " << count  << " inlier features in last frame associate with map points";
    //////////////////////////////////

    for (int i = 0; i < current_frame_->matches_with_last_frame_.size(); ++i) {
        cv::DMatch match = current_frame_->matches_with_last_frame_[i];

        auto current_feature = current_frame_->features_[match.trainIdx];
        auto last_feature = last_frame_->features_[match.queryIdx];

        if (last_feature->map_point_.lock() && (last_feature->is_inlier_ == true ||
                last_feature->associate_new_map_point_ == true)) {
            current_feature->map_point_ = last_feature->map_point_.lock();
            num_good_pts++;
        }
    }

    LOG(INFO) << "Find " << num_good_pts << " good 2D matches with last frame";
    return num_good_pts;
}

//int Frontend::TrackLastFrame()
//{
//    // use LK flow to estimate points in the right image
//    vector<cv::Point2f> kpts_last, kpts_current;
//    vector<std::shared_ptr<Feature>> last_frame_feature_inliers;
//    LOG(INFO) << last_frame_->features_.size();
//    for (auto &kp : last_frame_->features_) {
//        if (kp->map_point_.lock()) {
//            // use project point
//            auto mp = kp->map_point_.lock();
//            last_frame_feature_inliers.push_back(kp);
//            kpts_last.push_back(kp->position_.pt);
//        }
//    }
//    LOG(INFO) << last_frame_feature_inliers.size();
//
//    std::vector<uchar> status;
//    Mat error;
//    cv::calcOpticalFlowPyrLK(
//            last_frame_->rgb_img_, current_frame_->rgb_img_,
//            kpts_last, kpts_current, status, error);
//
//    //, cv::Size(11, 11), 3,
//    //            cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30,0.01),
//    //            cv::OPTFLOW_USE_INITIAL_FLOW
//
//    // need to compute descriptors for initialize features
//    vector<cv::KeyPoint> keypoints;
//    vector<int> indices;
//    cv::Mat descriptors;
//    int count = 0;
//    for (int i = 0; i < status.size(); i++) {
//        if (status[i]) {
//            cv::KeyPoint kp(kpts_current[i], 3);
//            keypoints.push_back(kp);
//            indices.push_back(i);
//            count++;
//        }
//    }
//
//    orb_->compute(current_frame_->rgb_img_, keypoints, descriptors);
//    cout << keypoints.size() << endl << descriptors.rows << endl;
//    for (int i = 0; i < keypoints.size(); i++) {
//        Feature::Ptr feature(new Feature(current_frame_, keypoints[i], descriptors.row(i)));
//        feature->map_point_ = last_frame_feature_inliers[indices[i]]->map_point_;
//        current_frame_->features_.push_back(feature);
//    }
//
//    int num_good_pts = keypoints.size();
//
//    int num_good_pts = 0;
//    for (int i = 0; i < status.size(); i++) {
//        if (status[i]) {
//            cv::KeyPoint kp(kpts_current[i], 7);
//            Feature::Ptr feature(new Feature(current_frame_, kp));
//            feature->map_point_ = last_frame_feature_inliers[i]->map_point_;
//            current_frame_->features_.push_back(feature);
//            num_good_pts++;
//        }
//    }
//
//    LOG(INFO) << "Find " << num_good_pts << " matches in last image.";
//
//    return num_good_pts;
//}


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
//    descriptors.convertTo(descriptors, CV_32F);

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

