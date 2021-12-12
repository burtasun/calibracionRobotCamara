#pragma once
using float_type = double;
#define MatType(r,c) using M##r##c = Eigen::Matrix<float_type,r,c>;
#define MatTypeSq(r) using M##r = Eigen::Matrix<float_type,r,r>;
#define MatTypeColsX(c) using M##c##x = Eigen::Matrix<float_type,c,-1>;
#define MatTypeRowsX(r) using M##x##r = Eigen::Matrix<float_type,-1,r>;
#define VecType(r) using V##r = Eigen::Matrix<float_type,r,1>;
#define VecTypeRow(c) using Vr##c = Eigen::Matrix<float_type,1,c>;
MatType(3, 4);
MatTypeSq(4);
MatTypeSq(3);
VecType(3);
VecType(4);
VecTypeRow(3);
VecTypeRow(4);
MatTypeColsX(2);
MatTypeColsX(3);
MatTypeRowsX(2);
MatTypeRowsX(3);
using Mx = Eigen::Matrix<float_type, -1, -1>;
#define M_PI 3.1415926535897932384626433832795
#define deg2rad(deg) (deg*M_PI/180.)
#define rad2deg(rad) (rad*180./M_PI)

//RT_CV
using RTCV = std::array<cv::Mat, 2>;

#define LOG
#define prt(var) std::cout<<#var<<": "<<var<<'\n';
