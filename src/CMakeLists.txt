add_library(myslam_lib SHARED
        config.cpp
        camera.cpp
        frame.cpp
        mappoint.cpp
        feature.cpp
        map.cpp
        dataset.cpp
        frontend.cpp
        utils.cpp
        backend.cpp
        algorithm.cpp ../include/myslam/visual_odometry.h visual_odometry.cpp)

target_link_libraries(myslam_lib
        ${THIRD_PARTY_LIBS})