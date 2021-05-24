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
#include "myslam/g2o_types.h"

namespace myslam {

Frontend::Frontend()
{
    static const int num_keypoints = Config::Get<int>("number_of_features");
    static const float scale_factor = Config::Get<float>("scale_factor");
    static const int level_pyramid = Config::Get<int>("level_pyramid");
    static const int score_threshold = Config::Get<int>("score_threshold");

    // -- Create ORB
    orb_ = cv::ORB::create(num_keypoints, scale_factor, level_pyramid,
                           31, 0, 2, cv::ORB::HARRIS_SCORE, 31, score_threshold);
}

bool Frontend::AddFrame(Frame::Ptr frame)
{
    current_frame_ = frame;

    frames_buff_.push_back(current_frame_);

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
    }

    if (last_frame_ != nullptr) {
        cv::Mat t = getPosFromT(current_frame_->Pose());
        cv::Mat t_relative = getPosFromT(relative_motion_);
        LOG(INFO) << "\nCamera motion:" << endl;
        LOG(INFO) << "Motion w.r.t World:\n " << current_frame_->Pose().inv() << endl;
//        LOG(INFO) << "Motion w.r.t last frame:\n " << t_relative << endl;
    }

    ///////////////////// Visualize 2 ////////////////////////////////
    vector<cv::KeyPoint> v_keypoints_map_proj, v_keypoints_2d;
    for (int i = 0; i < current_frame_->features_.size(); i++)
    {
        auto current_feature = current_frame_->features_[i];
        if (current_feature->map_point_.lock()) {
            cv::Point3f p_world = current_feature->map_point_.lock()->pos_;
            cv::Point2f p_img_proj = current_frame_->camera_->world2pixel(p_world, current_frame_->Pose());
            cv::Point2f p_img = current_feature->position_.pt;

            cv::circle(current_frame_->rgb_img_, p_img_proj, 3, cv::Scalar(0, 0, 255), cv::FILLED);
            cv::circle(current_frame_->rgb_img_, p_img, 3, cv::Scalar(0, 255, 0), cv::FILLED);
        }
    }

    cv::imshow ( "current", current_frame_->rgb_img_ );
    cv::waitKey(0);
    //////////////////////////////////////////////////////////////

    last_frame_ = current_frame_;

    return true;
}

bool Frontend::MonoInit()
{

    MatchFeatures(
            reference_frame_->features_, current_frame_->features_, current_frame_->matches_with_ref_frame_);
    LOG(INFO) << "Number of matches with the reference frame:" << current_frame_->matches_with_ref_frame_.size();

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
    TrackLastFrame();

    tracking_inliers_ = EstimateCurrentPoseByPNP();

    LOG(INFO) << "Tracking inliers: " << tracking_inliers_;

    if (tracking_inliers_ >= num_features_tracking_) {
        // tracking good
        status_ = FrontendStatus::TRACKING_GOOD;
    } else if (tracking_inliers_ >= num_features_tracking_bad_) {
        // tracking bad
        status_ = FrontendStatus::TRACKING_BAD;
    } else {
        // lost
        status_ = FrontendStatus::LOST;
    }

    if (status_ == FrontendStatus::TRACKING_GOOD || status_ == FrontendStatus::TRACKING_BAD) {
        InsertKeyframe();
    } else {
        Reset();
    }

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

    LocalBundleAdjustment();

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

    // Normalize t to set the distance as 1
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
    cv::Mat R, rvec, tvec;

    bool is_pnp_good = num_matches >= kMinPtsForPnP;
    int num_inliers = 0;
    if (is_pnp_good) {
        bool useExtrinsicGuess = false;      // provide initial guess
        int iterationsCount = 100;
        float reprojectionError = 2.0;
        double confidence = 0.999;
        cv::solvePnPRansac(pts_3d, pts_2d, camera_->K_, cv::Mat(), rvec, tvec,
                           useExtrinsicGuess,
                           iterationsCount, reprojectionError, confidence, pnp_inliers_mask, cv::SOLVEPNP_ITERATIVE);

        cv::Rodrigues(rvec, R); // angle-axis rotation to 3x3 rotation matrix
        pose_init_ = convertRt2T(R, tvec);

        PoseOptimization();

        for (int i = 0; i < pnp_inliers_mask.rows; i++)
        {
            int good_idx = pnp_inliers_mask.at<int>(i, 0);

            auto mappoint = features[good_idx]->map_point_.lock();
            auto p_proj = camera_->world2pixel(mappoint->pos_, pose_init_);
            auto p_img = features[good_idx]->position_.pt;

            float reprojectionError = (p_proj.x - p_img.x) * (p_proj.x - p_img.x) \
                                + (p_proj.y - p_img.y) * (p_proj.y - p_img.y);

            if (reprojectionError <= 4)
            {
                features[good_idx]->is_inlier_ = true;
                num_inliers++;
            }
        }

        if (num_inliers > num_features_tracking_bad_) {
            current_frame_->SetPose(pose_init_);
        }

    }

    return num_inliers;
}

void Frontend::LocalBundleAdjustment()
{
    // setup g2o
    typedef g2o::BlockSolver_6_3 BlockSolverType;
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>
            LinearSolverType;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<BlockSolverType>(
                    g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    // pose 顶点，使用 frame id
    std::map<unsigned long, g2o::VertexSE3Expmap*> vertices_poses;
    std::map<unsigned long, Frame::Ptr> frames;

    // K
    cv::Mat K = camera_->K_;

    // 卡方分布 95% 以上可信度的时候的阈值
    const float thHuber2D = sqrt(5.99);     // 自由度为2

    LOG(INFO) << "Add pose vertices";
    for (int i = 0; i < frames_buff_.size(); i++)
    {
        Frame::Ptr frame = frames_buff_[i];

        // pose vertex
        g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setId(frame->id_);
        vSE3->setEstimate(toSE3Quat(frame->Pose()));
        optimizer.addVertex(vSE3);

        vertices_poses.insert({frame->id_, vSE3});
        frames.insert({frame->id_, frame});
    }

    std::vector<g2o::EdgeSE3ProjectXYZOnlyPose *> edges;
    std::vector<Feature::Ptr> features;

    LOG(INFO) << "Add edges";
    int index = 1;
    for (int i = 0; i < frames_buff_.size(); i++)
    {
        Frame::Ptr frame = frames_buff_[i];

        // edges
        for (int j = 0; j < frame->features_.size(); j++) {
            auto feature = frame->features_[j];
            auto mp = feature->map_point_.lock();

            if (mp)
            {
                features.push_back(feature);

                g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();
                e->setId(index);
                e->setVertex(0, vertices_poses[frame->id_]);
                e->setMeasurement(Eigen::Vector2d(feature->position_.pt.x, feature->position_.pt.y));
                e->setInformation(Eigen::Matrix2d::Identity());

                // 使用鲁棒核函数
                auto rk = new g2o::RobustKernelHuber();
                rk->setDelta(thHuber2D);
                e->setRobustKernel(rk);

                // 设置相机内参
                e->fx = camera_->fx_;
                e->fy = camera_->fy_;
                e->cx = camera_->cx_;
                e->cy = camera_->cy_;

                // 地图点的空间位置,作为迭代的初始值
                e->Xw[0] = mp->pos_.x;
                e->Xw[1] = mp->pos_.y;
                e->Xw[2] = mp->pos_.z;

                optimizer.addEdge(e);

                edges.push_back(e);
                index++;
            }
        }
    }

    // do optimization and eliminate the outliers
    optimizer.initializeOptimization();
    optimizer.optimize(20);

    // count the outliers
    int cnt_outliers = 0;
    for (int i = 0; i < edges.size(); i++)
    {
        auto e = edges[i];

        e->computeError();
        Eigen::Matrix<double, 2, 1> error = e->error();
//        LOG(INFO) << error(0, 0) << '\t' << error(1, 0);
        if (e->chi2() > 5.991) {
            features[i]->is_inlier_ = false;
            cnt_outliers++;
        }
    }

    LOG(INFO) << "Outlier/Inlier in pose estimating: " << cnt_outliers << "/"
              << features.size() - cnt_outliers;

    // Set pose and lanrmark position
    for (auto &v : vertices_poses) {
        frames[v.first]->SetPose(toCvMat(v.second->estimate()));
    }

    for (auto &feat : features) {
        if (!feat->is_inlier_) {
            feat->map_point_.reset(); // maybe we can still use it in future
        }
    }

    // Reset frames buffer
    frames_buff_.clear();
    frames_buff_.push_back(current_frame_);
}

// 以 PnP 的结果为初始化, 优化当前帧的 Pose
void Frontend::PoseOptimization()
{
    // setup g2o
    typedef g2o::BlockSolver_6_3 BlockSolverType;
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<BlockSolverType>(
                    g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    // K
    cv::Mat K = camera_->K_;

    // 卡方分布 95% 以上可信度的时候的阈值
    const float thHuber2D = sqrt(5.99);     // 自由度为2

    // pose vertex
    g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
    vSE3->setId(0);
    vSE3->setEstimate(toSE3Quat(pose_init_));
    optimizer.addVertex(vSE3);

    // edges
    int index = 0;
    for (int i = 0; i < current_frame_->features_.size(); i++) {
        auto feature = current_frame_->features_[i];
        auto mp = feature->map_point_.lock();

        if (mp)
        {
            g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();
            e->setId(index);
            e->setVertex(0, vSE3);
            e->setMeasurement(Eigen::Vector2d(feature->position_.pt.x, feature->position_.pt.y));
            e->setInformation(Eigen::Matrix2d::Identity());

            // 使用鲁棒核函数
            auto rk = new g2o::RobustKernelHuber();
            rk->setDelta(thHuber2D);
            e->setRobustKernel(rk);

            // 设置相机内参
            e->fx = camera_->fx_;
            e->fy = camera_->fy_;
            e->cx = camera_->cx_;
            e->cy = camera_->cy_;

            // 地图点的空间位置,作为迭代的初始值
            e->Xw[0] = mp->pos_.x;
            e->Xw[1] = mp->pos_.y;
            e->Xw[2] = mp->pos_.z;

            optimizer.addEdge(e);
            index++;
        }
    }

    // do optimization and eliminate the outliers
    optimizer.initializeOptimization();
    optimizer.optimize(10);

    pose_init_ = toCvMat(vSE3->estimate());

}

int Frontend::TrackLastFrame()
{
    int num_good_pts = 0;

    // Descriptor tracking
    MatchFeatures(
            last_frame_->features_, current_frame_->features_, current_frame_->matches_with_last_frame_);

    for (int i = 0; i < current_frame_->matches_with_last_frame_.size(); ++i) {
        cv::DMatch match = current_frame_->matches_with_last_frame_[i];

        auto current_feature = current_frame_->features_[match.trainIdx];
        auto last_feature = last_frame_->features_[match.queryIdx];

        if (last_feature->map_point_.lock() &&
            (last_feature->is_inlier_ || last_feature->associate_new_map_point_))
        {
            current_feature->map_point_ = last_feature->map_point_.lock();
            num_good_pts++;
        }
    }

    LOG(INFO) << "Find " << num_good_pts << " good 2D matches with last frame";
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
