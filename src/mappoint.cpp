//
// Created by jihaorui on 5/1/21.
//

#include "myslam/mappoint.h"
#include "myslam/feature.h"

namespace myslam {

    MapPoint::MapPoint(long id, cv::Point3f position) : id_(id), pos_(position) {}

    MapPoint::Ptr MapPoint::CreateNewMappoint() {
        static long factory_id = 0;
        MapPoint::Ptr new_mappoint(new MapPoint);
        new_mappoint->id_ = factory_id++;
        return new_mappoint;
    }

    void MapPoint::RemoveObservation(std::shared_ptr<Feature> feat) {
        for (auto iter = observations_.begin(); iter != observations_.end();
             iter++) {
            if (iter->lock() == feat) {
                observations_.erase(iter);
                feat->map_point_.reset();
                observed_times_--;
                break;
            }
        }
    }


}  // namespace myslam
