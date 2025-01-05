#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

struct p2pFactor // 点到点的残差距离计算，这里也可以使用class，不一定是struct
{
	p2pFactor(Eigen::Vector3d vertex_frame1, Eigen::Vector3d vertex_frame2, double weight)
		: vertex_frame1_(vertex_frame1), vertex_frame2_(vertex_frame2), weight_(weight){}
	// curr_point是当前点; last_point_a是上一帧的点a; last_point_b是上一帧的点b; s是当前点的时间戳占比

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const // q是优化变量四元数，t是优化变量平移部分，residual是残差模块
	{

		Eigen::Matrix<T, 3, 1> point1_mat{T(vertex_frame1_.x()), T(vertex_frame1_.y()), T(vertex_frame1_.z())};                 // 模版函数，数据类型转换
		Eigen::Matrix<T, 3, 1> point2_mat{T(vertex_frame2_.x()), T(vertex_frame2_.y()), T(vertex_frame2_.z())};                 // 将double类型的curr_point.x()转成T类型存储在Matrix里

		Eigen::Quaternion<T> q_frame2_to_frame1{q[3], q[0], q[1], q[2]};      //order: wxyz
		Eigen::Matrix<T, 3, 1> t_frame2_to_frame1{t[0], t[1], t[2]};          //order: xyz


        Eigen::Matrix<T, 3, 1> point_frame2_in_frame1;                 //映射后的值   
        point_frame2_in_frame1 = q_frame2_to_frame1 * point2_mat + t_frame2_to_frame1; //lidar vertex 坐标映射到卫星图上
        
        //differences
        T x_diff =  point_frame2_in_frame1.x() - point1_mat.x();
        T y_diff =  point_frame2_in_frame1.y() - point1_mat.y();
        T z_diff =  point_frame2_in_frame1.z() - point1_mat.z();
        //二范数残差
        residual[0] = sqrt(x_diff*x_diff + y_diff*y_diff + z_diff*z_diff) * weight_;

		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d vertex_frame1, const Eigen::Vector3d vertex_frame2, double weight)
	{
		return (new ceres::AutoDiffCostFunction< // 自动导数(AutoDiffCostFunction): 由ceres自行决定导数的计算方式，最常用的求导方式
				p2pFactor, 1, 4, 3>( // 仿函数类型CostFunctor，残差维数1，四元数维数4，平移向量维数3 (仿函数类内部必须重载了操作符()。)
			    new p2pFactor(vertex_frame1, vertex_frame2, weight)));
	}

	Eigen::Vector3d vertex_frame1_, vertex_frame2_;
	double weight_;
};
