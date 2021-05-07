//
// Created by jihaorui on 5/6/21.
//

#ifndef MYSLAM_BACKEND_H
#define MYSLAM_BACKEND_H

#include "myslam/common_include.h"
#include "myslam/frame.h"
#include "myslam/map.h"

namespace myslam
{

class Map;
/**
 * 后端
 * 有单独优化线程，在Map更新时启动优化
 * Map更新由前端触发
 */
class Backend
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Backend> Ptr;

    /// 构造函数中启动优化线程并挂起
    Backend();

    // 设置相机，用于获得内外参
    void SetCameras(Camera::Ptr cam) {
        camera_ = cam;
    }

    /// 设置地图
    void SetMap(std::shared_ptr<Map> map) { map_ = map; }

    /// 触发地图更新，启动优化
    void UpdateMap();

private:

    /// 对给定关键帧和路标点进行优化
    void Optimize(Map::KeyframesType& keyframes, Map::LandmarksType& landmarks);

    std::shared_ptr<Map> map_;

    Camera::Ptr camera_ = nullptr;
};

}  // namespace myslam

#endif  // MYSLAM_BACKEND_H
