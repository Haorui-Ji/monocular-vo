//
// Created by jihaorui on 5/18/21.
//

#ifndef MYSLAM_G2O_TYPES_H
#define MYSLAM_G2O_TYPES_H

#include "myslam/common_include.h"

#include <opencv2/core/eigen.hpp>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

#include <g2o/types/sim3/types_seven_dof_expmap.h>
#include <g2o/types/sba/types_sba.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

namespace myslam
{
    //将变换矩阵转换为李代数se3：cv:Mat->g2o::SE3Quat
    g2o::SE3Quat toSE3Quat(const cv::Mat &cvT)
    {
        //首先将旋转矩阵提取出来
        Eigen::Matrix<double,3,3> R;
        R << cvT.at<double>(0,0), cvT.at<double>(0,1), cvT.at<double>(0,2),
                cvT.at<double>(1,0), cvT.at<double>(1,1), cvT.at<double>(1,2),
                cvT.at<double>(2,0), cvT.at<double>(2,1), cvT.at<double>(2,2);

        //然后将平移向量提取出来
        Eigen::Matrix<double,3,1> t(cvT.at<double>(0,3), cvT.at<double>(1,3), cvT.at<double>(2,3));

        //构造g2o::SE3Quat类型并返回
        return g2o::SE3Quat(R,t);
    }

    //李代数se3转换为变换矩阵：g2o::SE3Quat->cv::Mat
    cv::Mat toCvMat(const g2o::SE3Quat &SE3)
    {
        ///在实际操作上，首先转化成为Eigen中的矩阵形式，然后转换成为cv::Mat的矩阵形式。
        Eigen::Matrix<double,4,4> eigMat = SE3.to_homogeneous_matrix();
        //然后再由Eigen::Matrix->cv::Mat
        //首先定义存储计算结果的变量
        cv::Mat cvMat(4,4,CV_64F);
        //然后逐个元素赋值
        for(int i=0;i<4;i++)
            for(int j=0; j<4; j++)
                cvMat.at<double>(i,j)=eigMat(i,j);

        //返回计算结果，还是用深拷贝函数
        return cvMat.clone();
    }

    // 将OpenCV中Mat类型的向量转化为Eigen中Matrix类型的变量
    Eigen::Matrix<double,3,1> toVector3d(const cv::Mat &cvVector)
    {
        //首先生成用于存储转换结果的向量
        Eigen::Matrix<double,3,1> v;
        //然后通过逐个赋值的方法完成转换
        v << cvVector.at<float>(0), cvVector.at<float>(1), cvVector.at<float>(2);
        //返回转换结果
        return v;
    }

    // 将OpenCV中cv::Point3f类型的向量转化为Eigen中Matrix类型的变量
    Eigen::Matrix<double,3,1> toVector3d(const cv::Point3f &cvPoint)
    {
        //首先生成用于存储转换结果的向量
        cv::Mat vMat = (cv::Mat_<double>(3, 1) << cvPoint.x, cvPoint.y, cvPoint.z);
        return toVector3d(vMat);
    }

    // 将Eigen中Matrix类型的变量转化为OpenCV中cv::Point3f类型的变量
    cv::Point3f toPoint3f(const Eigen::Matrix<double,3,1> &vec)
    {
        //首先生成用于存储转换结果的向量
        cv::Point3f cvPoint(
                vec(0, 0),
                vec(1, 0),
                vec(2, 0));
        return cvPoint;
    }

}  // namespace myslam

#endif  // MYSLAM_G2O_TYPES_H

