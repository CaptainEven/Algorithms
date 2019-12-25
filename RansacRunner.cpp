#include "RansacRunner.h"
#include<iostream>

RansacRunner::RansacRunner(const float tolerance,
	const int num_sample,
	const int num_sub_sample,
	const int num_iter)
{
	this->m_tolerance = tolerance;
	this->m_num_iter = num_iter;
	this->m_num_sample = num_sample;
	this->m_num_sub_sample = num_sub_sample;

	// 预先开辟内存
	if (this->m_num_sub_sample > 3)
	{
		this->m_subsets.reserve(this->m_num_sub_sample);
		this->m_subsets.resize(this->m_num_sub_sample);
	}

	this->m_num_inliers = 0;

	memset(this->m_plane, 0.0f, sizeof(float) * 4);
}

int RansacRunner::GetMinSubSets(const std::vector<cv::Point3f>& Pts3D)
{
	int id_0 = rand() % int(this->m_num_sample);
	int id_1 = rand() % int(this->m_num_sample);
	int id_2 = rand() % int(this->m_num_sample);

	this->m_min_subsets[0] = Pts3D[id_0];
	this->m_min_subsets[1] = Pts3D[id_1];
	this->m_min_subsets[2] = Pts3D[id_2];

	return 0;
}

int RansacRunner::GetSubSets(const std::vector<cv::Point3f>& Pts3D)
{
	for (int i = 0; i < this->m_num_sub_sample; ++i)
	{
		int id = rand() % int(this->m_num_sample);
		this->m_subsets[i] = Pts3D[id];
	}

	return 0;
}

int RansacRunner::UpdateNumIters(double p, double ep, int modelPoints, int maxIters)
{
	if (modelPoints <= 0)
	{
		printf("[Err]: the number of model points should be positive");
	}

	p = MAX(p, 0.);
	p = MIN(p, 1.);
	ep = MAX(ep, 0.);
	ep = MIN(ep, 1.);

	// avoid inf's & nan's
	double num = MAX(1. - p, DBL_MIN);
	double denom = 1. - std::pow(1. - ep, modelPoints);
	if (denom < DBL_MIN)
	{
		return 0;
	}

	num = std::log(num);
	denom = std::log(denom);

	return denom >= 0 || -num >= maxIters * (-denom) ? maxIters : cvRound(num / denom);
}

int RansacRunner::PlaneFitBy3Pts(const cv::Point3f* pts, float* plane_arr)
{
	const float& x1 = pts[0].x;
	const float& y1 = pts[0].y;
	const float& z1 = pts[0].z;

	const float& x2 = pts[1].x;
	const float& y2 = pts[1].y;
	const float& z2 = pts[1].z;

	const float& x3 = pts[2].x;
	const float& y3 = pts[2].y;
	const float& z3 = pts[2].z;

	float A = (y2 - y1)*(z3 - z1) - (y3 - y1)*(z2 - z1);
	float B = (z2 - z1)*(x3 - x1) - (z3 - z1)*(x2 - x1);
	float C = (x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1);

	const float DENOM = std::sqrtf(A * A + B * B + C * C);

	// 判断三点是否共线
	if (DENOM < 1e-12)
	{
		//printf("[Warning]: 3 Points may near colinear\n");
		return -1;
	}

	A /= DENOM;
	B /= DENOM;
	C /= DENOM;
	float D = -(A*x1 + B * y1 + C * z1);

	plane_arr[0] = A;
	plane_arr[1] = B;
	plane_arr[2] = C;
	plane_arr[3] = D;

	return 0;
}

// 3D平面方程拟合(最小二乘法)写成Ax=B的形式: aX + bY + Z + c = 0(aX + bY + c = -Z)
int RansacRunner::PlaneFitOLS(float* plane_arr)
{// 输出结果向量X: a, b, c
	cv::Mat A(this->m_num_sub_sample, 3, CV_32F);
	cv::Mat B(this->m_num_sub_sample, 1, CV_32F);

	// 系数矩阵A和结果向量b初始化
	for (size_t i = 0; i < this->m_num_sub_sample; ++i)
	{// 
		A.at<float>((int)i, 0) = this->m_subsets[i].x;
		A.at<float>((int)i, 1) = this->m_subsets[i].y;
		A.at<float>((int)i, 2) = 1.0f;

		B.at<float>((int)i, 0) = -this->m_subsets[i].z;
	}

	// 解线性方程组: x = (A' * A)^-1 * A' * b
	cv::Mat X = -((A.t() * A).inv() * A.t() * B);  // 3×1

	const float& a = X.at<float>(0, 0);
	const float& b = X.at<float>(1, 0);
	const float& c = X.at<float>(2, 0);
	const float DENOM = std::sqrtf(a * a + b * b + 1.0f);

	X.at<float>(0, 0) /= DENOM;  // a
	X.at<float>(1, 0) /= DENOM;  // b
	X.at<float>(2, 0) /= DENOM;  // c

	plane_arr[0] = X.at<float>(0, 0);
	plane_arr[1] = X.at<float>(1, 0);
	plane_arr[2] = 1.0f / DENOM;
	plane_arr[3] = X.at<float>(2, 0);

	return 0;
}

int RansacRunner::CountInliers(const float* plane_arr, 
	const std::vector<cv::Point3f>& Pts3D)
{
	cv::Mat plane_mat(4, 1, CV_32F);
	plane_mat.at<float>(0, 0) = plane_arr[0];
	plane_mat.at<float>(1, 0) = plane_arr[1];
	plane_mat.at<float>(2, 0) = plane_arr[2];
	plane_mat.at<float>(3, 0) = plane_arr[3];

	int count = 0;
	cv::Mat point_mat(4, 1, CV_32F);
	for (auto pt : Pts3D)
	{
		point_mat.at<float>(0, 0) = pt.x;
		point_mat.at<float>(1, 0) = pt.y;
		point_mat.at<float>(2, 0) = pt.z;
		point_mat.at<float>(3, 0) = 1.0f;

		float dist = fabs((float)plane_mat.dot(point_mat)) \
			/ sqrtf(plane_arr[0] * plane_arr[0] + \
				plane_arr[1] * plane_arr[1] + plane_arr[2] * plane_arr[2]);
		if (dist < m_tolerance)
		{
			count++;
		}
	}

	if (count > this->m_num_inliers)
	{
		// update inlier number
		this->m_num_inliers = count;

		// update plane 's 4 parameters
		memcpy(m_plane, plane_arr, sizeof(float) * 4);
	}

	return 0;
}

// 运行RANSAC迭代
int RansacRunner::RunRansac(const std::vector<cv::Point3f>& Pts3D)
{
	if (this->m_num_sub_sample < 3)
	{
		printf("[Err]: not enough sub_sample_num\n");
		return -1;
	}

	float plane_arr[4];
	for (int i = 0; i < this->m_num_iter; i++)
	{
		// 取样本子集
		if (3 == this->m_num_sub_sample)
		{
			this->GetMinSubSets(Pts3D);
		}
		else
		{
			this->GetSubSets(Pts3D);
		}

		// 计算平面方程存于plane_arr
		if (3 == this->m_num_sub_sample)
		{
			int ret = this->PlaneFitBy3Pts(this->m_min_subsets, plane_arr);
			if (-1 == ret)
			{
				continue;
			}
		}
		else
		{
			this->PlaneFitOLS(plane_arr);
		}

		// 统计Inliers
		this->CountInliers(plane_arr, Pts3D);

		// 根据Outlier ratio更新总的迭代次数
		//int pre_num_iter = this->m_num_iter;
		this->m_num_iter = this->UpdateNumIters(0.9985f, 
			(double)((int)Pts3D.size() - this->m_num_inliers) / (double)Pts3D.size(),
			this->m_num_sub_sample, 
			this->m_num_iter);
		//printf("Update num_iter from %d to %d\n", pre_num_iter, this->m_num_iter);
	}

	//printf("Ransac done\n");
	return 0;
}

RansacRunner::~RansacRunner()
{

}
