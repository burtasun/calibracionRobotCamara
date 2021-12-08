#pragma once
using float_t = float;
#define MatType(r,c) using M##r##c = Eigen::Matrix<float_t,r,c>;
#define MatTypeSq(r) using M##r = Eigen::Matrix<float_t,r,r>;
#define MatTypeColsX(c) using M##c##x = Eigen::Matrix<float_t,c,-1>;
#define MatTypeRowsX(r) using M##x##r = Eigen::Matrix<float_t,-1,r>;
#define VecType(r) using V##r = Eigen::Matrix<float_t,r,1>;
#define VecTypeRow(c) using Vr##r = Eigen::Matrix<float_t,1,c>;
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
