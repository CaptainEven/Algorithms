#ifndef RANSAC_RUNNER
#define RANSAC_RUNNER

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/base.hpp>

class RansacRunner
{
public:
	// 初始化
	RansacRunner(const float tolerance,
		const int num_sample,
		const int num_sub_sample,
		const int num_iter=2000);

	// 取样本子集: 3个样本点
	int GetMinSubSets(const std::vector<cv::Point3f>& Pts3D);

	// 取样本子集: 多于3个样本点
	int GetSubSets(const std::vector<cv::Point3f>& Pts3D);

	// 根据outlier ratio更新总的迭代次数
	int UpdateNumIters(double p, double ep, int modelPoints, int maxIters);

	// 3个空间点(非共线)确定一个空间平面
	int PlaneFitBy3Pts(const cv::Point3f* pts, float* plane_arr);

	// 3D平面方程拟合(最小二乘法)写成Ax=B的形式: aX + bY + Z + c = 0(aX + bY + c = -Z)
	int PlaneFitOLS(float* plane_arr);

	// 统计内点(inlier)个数
	int CountInliers(const float* plane_arr, const std::vector<cv::Point3f>& Pts3D);

	// 运行RANSAC
	int RunRansac(const std::vector<cv::Point3f>& Pts3D);

	~RansacRunner();

//private:
	float m_tolerance;
	int m_num_sample;
	int m_num_sub_sample;
	int m_num_iter;
	int m_num_inliers;

	float m_plane[4];
	cv::Point3f m_min_subsets[3];
	std::vector<cv::Point3f> m_subsets;
};

#endif // !1