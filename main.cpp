#define _CRT_SECURE_NO_WARNINGS

#include<iostream>
#include<fstream>
#include<math.h>

#include<opencv2/opencv.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/ximgproc.hpp>

#include"guidedfilter.h"
#include "main.h"


using namespace std;
using namespace cv;


// ---------------- Functions
Mat GuideFilter(Mat& src, Mat& guide, int radius, double eps)
{
	// 64-float: to compute more accurately
	src.convertTo(src, CV_64FC1);
	guide.convertTo(guide, CV_64FC1);

	Mat mean_p, mean_I, mean_Ip, mean_II;

	boxFilter(src, mean_p, CV_64FC1, Size(radius, radius));
	boxFilter(guide, mean_I, CV_64FC1, Size(radius, radius));
	boxFilter(src.mul(guide), mean_Ip, CV_64FC1, Size(radius, radius));
	boxFilter(guide.mul(guide), mean_II, CV_64FC1, Size(radius, radius));

	// compute NCC
	Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);
	Mat var_I = mean_II - mean_I.mul(mean_I);

	// coefficients: a, b
	Mat a = cov_Ip / (var_I + eps);
	Mat b = mean_p - a.mul(mean_I);

	// compute mena of a, b
	Mat mean_a, mean_b;
	boxFilter(a, mean_a, CV_64FC1, Size(radius, radius));
	boxFilter(b, mean_b, CV_64FC1, Size(radius, radius));

	Mat dst = mean_a.mul(src) + mean_b;
	return dst;
}

int TestGuideFilter(uint mode=0)
{
	const auto path_src = string("./depth.jpg");
	const auto path_guide = string("./src.jpg");

	Mat src = imread(path_src, cv::IMREAD_GRAYSCALE);
	Mat guide = imread(path_guide, cv::IMREAD_GRAYSCALE);
	if (src.empty() || guide.empty())
	{
		cout << "[Err]: empty image." << endl;
		return -1;
	}

	Mat dst(src.rows, src.cols, src.type());
	int k = 10;

	// ------------ 用原图(RGB or gray?)引导深度图
	uint win_size = 2 * k - 1;
	cout << "Filter window size: " << win_size << "×" << win_size << endl;

	float eta = float(1e-6 * 255.0 * 255.0);
	dst = GuideFilter(src, guide, win_size, eta);

	// ------------ 联合双边滤波
	//ximgproc::jointBilateralFilter(guide, src, dst, -1, 2*k-1, 2*k - 1);

	// 转换引导后的图像到原图类型
	dst.convertTo(dst, CV_8UC1);

	if (!dst.empty())
	{
		std::string();
		char guide_filter_path[100];
		sprintf(guide_filter_path, "./guide_filter_%d×%d.jpg", win_size, win_size);
		imwrite(guide_filter_path, dst);

		Mat dst_;
		resize(dst, dst_,
			Size(uint(src.cols * 0.4f), uint(src.rows * 0.4f)),
			INTER_CUBIC);

		cv::imshow(guide_filter_path, dst_);
		cv::waitKey();
	}

	return 0;
}

double CalcEdSum(const int& HEIGHT, const int& WIDTH,
	const cv::Mat& sobel_xx, const cv::Mat& sobel_yy,
	const cv::Mat& depth_map, const cv::Mat& mat)
{
	const double sigma_1 = 0.1, alpha = 50.0;
	double Ed_sum = 0.0,
		d2_deriv_x = 0.0, d2_deriv_y = 0.0,
		w_x = 0.0, w_y = 0.0;
	for (int y = 0; y < HEIGHT; ++y)
	{
		for (int x = 0; x < WIDTH; ++x)
		{
			if (x < 1 || y < 1 ||
				x >= WIDTH - 1 || y >= HEIGHT - 1)
			{
				continue;
			}

			d2_deriv_x = 2.0 * mat.at<double>(y, x) - mat.at<double>(y, x - 1) - mat.at<double>(y, x + 1);
			d2_deriv_y = 2.0 * mat.at<double>(y, x) - mat.at<double>(y - 1, x) - mat.at<double>(y + 1, x);

			w_x = exp(-1.0 * abs(sobel_xx.at<double>(y, x) / sigma_1));
			w_y = exp(-1.0 * abs(sobel_yy.at<double>(y, x) / sigma_1));

			Ed_sum += pow(depth_map.at<double>(y, x) - mat.at<double>(y, x), 2) + alpha * (w_x * pow(d2_deriv_x, 2) + w_y * pow(d2_deriv_y, 2));
		}
	}

	return Ed_sum;
}

void propagate(const Mat& depth_map, const int& x, const int& y,
	const Mat& sobel_xx, const Mat& sobel_yy,
	const int& HEIGHT, const int& WIDTH,
	const int is_even,
	Mat& mat,
	double& Ed_sum)
{
	if (is_even)
	{
		// 计算right, down neighbor能量
		Mat* mat_right = new Mat(mat);
		Mat* mat_down = new Mat(mat);

		(*mat_right).at<double>(y, x) = mat.at<double>(y, x + 1);
		(*mat_down).at<double>(y, x) = mat.at<double>(y + 1, x);

		double Ed_right = CalcEdSum(HEIGHT, WIDTH,
									sobel_xx, sobel_yy,
									depth_map, *mat_right);
		double Ed_down = CalcEdSum(HEIGHT, WIDTH,
								   sobel_xx, sobel_yy,
								   depth_map, *mat_down);

		double Ed_min = min(Ed_right, Ed_down);
		if (Ed_min < Ed_sum)
		{
			if (Ed_min == Ed_right)
			{
				mat.at<double>(y, x) = mat.at<double>(y, x + 1);
				Ed_sum = Ed_right;
				printf("Update Ed_sum @(%d, %d) | Ed_sum: %.3f\n", x, y, Ed_sum);
			}
			else if (Ed_min == Ed_down)
			{
				mat.at<double>(y, x) = mat.at<double>(y + 1, x);
				Ed_sum = Ed_down;
				printf("Update Ed_sum @(%d, %d) | Ed_sum: %.3f\n", x, y, Ed_sum);
			}

			delete mat_right, mat_down;
			mat_right = nullptr;
			mat_down = nullptr;
		}
		else
		{
			delete mat_right, mat_down;
			mat_right = nullptr;
			mat_down = nullptr;
		}
	}
	else
	{
		// 计算right, down neighbor能量
		Mat* mat_left = new Mat(mat);
		Mat* mat_up = new Mat(mat);

		(*mat_left).at<double>(y, x) = mat.at<double>(y, x - 1);
		(*mat_up).at<double>(y, x) = mat.at<double>(y - 1, x);

		double Ed_left = CalcEdSum(HEIGHT, WIDTH,
								   sobel_xx, sobel_yy,
								   depth_map, *mat_left);
		double Ed_up = CalcEdSum(HEIGHT, WIDTH,
								 sobel_xx, sobel_yy,
								 depth_map, *mat_up);

		double Ed_min = min(Ed_left, Ed_up);
		if (Ed_min < Ed_sum)
		{
			if (Ed_min == Ed_left)
			{
				mat.at<double>(y, x) = mat.at<double>(y, x - 1);
				Ed_sum = Ed_left;
				printf("Update Ed_sum @(%d, %d) | Ed_sum: %.3f\n", x, y, Ed_sum);
			}
			else if (Ed_min == Ed_up)
			{
				mat.at<double>(y, x) = mat.at<double>(y - 1, x);
				Ed_sum = Ed_up;
				printf("Update Ed_sum @(%d, %d) | Ed_sum: %.3f\n", x, y, Ed_sum);
			}

			delete mat_left, mat_up;
			mat_left = nullptr;
			mat_up = nullptr;
		}
		else
		{
			delete mat_left, mat_up;
			mat_left = nullptr;
			mat_up = nullptr;
		}
	}
}

// 随机搜索
void random_search(const Mat& depth_map, const int& x, const int& y,
	const Mat& sobel_xx, const Mat& sobel_yy,
	const int& HEIGHT, const int& WIDTH,
	Mat& mat,
	double& Ed_sum)
{
	cv::RNG rng;
	double new_depth = rng.uniform(0.0, 255.0);

	Mat* mat_new = new Mat(mat);
	(*mat_new).at<double>(y, x) = new_depth;

	double Ed_new = CalcEdSum(HEIGHT, WIDTH, sobel_xx, sobel_yy, depth_map, *mat_new);

	if (Ed_new < Ed_sum)
	{
		mat.at<double>(y, x) = new_depth;
		Ed_sum = Ed_new;
		printf("Update Ed_sum @(%d, %d) | Ed_sum: %.3f\n", x, y, Ed_sum);
	}

	delete mat_new;
	mat_new = nullptr;
}

void TestPatchMatchOptimize()
{
	const auto path_depth = string("./depth.jpg");
	const auto path_src = string("./src.jpg");

	Mat depth_map = imread(path_depth, cv::IMREAD_GRAYSCALE);
	Mat src = imread(path_src, cv::IMREAD_GRAYSCALE);
	if (depth_map.empty() || src.empty())
	{
		cout << "[Err]: empty image." << endl;
		return;
	}
	assert(depth_map.cols == src.cols && depth_map.rows == src.rows);

	const int& WIDTH = depth_map.cols;
	const int& HEIGHT = depth_map.rows;

	// 计算一阶, 二阶sobel
	Mat sobel_x, sobel_y, sobel_xx, sobel_yy;
	Sobel(src, sobel_x, CV_16S, 1, 0, 3);
	Sobel(src, sobel_y, CV_16S, 0, 1, 3);
	Sobel(sobel_x, sobel_xx, CV_16S, 1, 0, 3);
	Sobel(sobel_y, sobel_yy, CV_16S, 0, 1, 3);

	// 计算初始估计深度图
	cv::RNG rng;
	Mat mat = Mat::zeros(HEIGHT, WIDTH, CV_64FC1);
	for (int i = 0; i < HEIGHT; ++i)
	{
		for (int j = 0; j < WIDTH; ++j)
		{
			mat.at<double>(i, j) = rng.uniform(0.0, 255.0);
		}
	}

	// 将数据类型转换为double
	src.convertTo(src, CV_64FC1);
	depth_map.convertTo(depth_map, CV_64FC1);
	sobel_xx.convertTo(sobel_xx, CV_64FC1);
	sobel_yy.convertTo(sobel_yy, CV_64FC1);

	// 计算初始能量
	double Ed_sum = CalcEdSum(HEIGHT, WIDTH, sobel_xx, sobel_yy, depth_map, mat);
	printf("=> Initial Ed sum: %.3f\n", Ed_sum);

	// 开始迭代优化...
	const int NUM_ITER = 50;
	for (int iter_i = 0; iter_i < NUM_ITER; ++iter_i)
	{
		int is_even = iter_i % 2 - 1;
		if (is_even)
		{
			// check right, down neighbor
			for (int y = 0; y < HEIGHT - 1; ++y)
			{
				for (int x = 0; x < WIDTH - 1; ++x)
				{
					// 传播
					propagate(depth_map, x, y, 
						sobel_xx, sobel_yy, 
						HEIGHT, WIDTH, 
						is_even, mat, Ed_sum);

					// 随机搜索
					random_search(depth_map, x, y,
						sobel_xx, sobel_yy,
						HEIGHT, WIDTH,
						mat, Ed_sum);
				}
			}
		}
		else  
		{
			// check left, up neighbor
			for (int y = HEIGHT - 1; y > 0; --y)
			{
				for (int x = WIDTH - 1; x > 0; --x)
				{
					// 传播
					propagate(depth_map, x, y,
						sobel_xx, sobel_yy, 
						HEIGHT, WIDTH,
						is_even, mat, Ed_sum);

					// 随机搜索
					random_search(depth_map, x, y,
						sobel_xx, sobel_yy, 
						HEIGHT, WIDTH, 
						mat, Ed_sum);
				}
			}
		}

		printf("Iter %d | Ed_sum: %.3f\n", iter_i, Ed_sum);

		// 保存中间结果
		Mat output(HEIGHT, WIDTH, CV_8UC1);
		mat.convertTo(output, CV_8UC1);

		char buff[100];
		sprintf(buff, "Iter_%d.jpg", iter_i + 1);

		imwrite(buff, output);
	}

	system("pause");
}

void TestMatVector()
{
	int a[5] = { 1, 3, 5, 7, 5 };

	std::vector<int> data;
	for (int i = 0; i < 5; ++i)
	{
		data.push_back(a[i]);
	}
	for (int i = 0; i < 5; ++i)
	{
		printf("%d ", data[i]);
	}
}

// 测试opencv表征高维数组(Tensor)
void TestDepthSR()
{
	const string src_path("./src4.jpg");
	const string depth_path("./depth4.jpg");

	cv::Mat src = cv::imread(src_path, cv::IMREAD_GRAYSCALE);
	cv::Mat depth_map = cv::imread(depth_path, cv::IMREAD_GRAYSCALE);

	if (src.empty() || depth_map.empty())
	{
		printf("[Err]: empty image.\n");
		return;
	}

	assert(src.cols == depth_map.cols && src.rows == depth_map.rows);

	const int HEIGHT = src.rows, WIDTH = src.cols;
	printf("HEIGHT: %d, WIDTH: %d\n", HEIGHT, WIDTH);

	// 原图运算LOG算子
	//cv::Mat blur, joint;
	//cv::GaussianBlur(src, blur, cv::Size(3, 3), 0);
	//cv::Laplacian(blur, joint, CV_16S, 3);
	cv::Mat joint = src;

	// 图像数据转换为float32
	joint.convertTo(joint, CV_32FC1);
	depth_map.convertTo(depth_map, CV_32FC1);

	const int depth_hypos = 256;

	// 迭代优化depthmap
	float eta = 0.5f, L = float(depth_hypos);
	float sigma_c = 10.0f, sigma_s = 10.0f;
	const float THRESH = eta * L;
	printf("THRESH: %.3f\n", THRESH);
	int num_iter = 5;

	for (int iter_i = 0; iter_i < num_iter; ++iter_i)
	{
		// 构建cost_volume,并计算cost_cw
		//cv::Mat costs_filtered[depth_hypos];
		cv::Mat* costs_filtered = new cv::Mat[depth_hypos];

		// 遍历每个depth htpothesis
		for (int d = 0; d < depth_hypos; ++d)
		{
			// rows, cols, type, scalar
			cv::Mat cost(HEIGHT, WIDTH, CV_32FC1, float(d));
			cost -= depth_map;

			cv::pow(cost, 2, cost);

			// 对cost进行clamp
			cv::min(THRESH, cost, cost);

			// 调用联合双边滤波
			cv::Mat dst;
			cv::ximgproc::jointBilateralFilter(joint,
											   cost,
											   dst,
										       -1, sigma_c, sigma_s,
										       cv::BORDER_DEFAULT);
			costs_filtered[d] = dst;
			printf("Depth hypothesis %d cost filtered\n", d);
		}

		// 更新depthmap
		cv::Mat dm_tmp(HEIGHT, WIDTH, CV_32FC1);
		for (int y = 0; y < HEIGHT; ++y)
		{
			for (int x = 0; x < WIDTH; ++x)
			{
				// 搜索每一个坐标(x, y)cost最小的depth(idx)
				float min_cost = FLT_MAX;
				int min_idx = -1;
				for (int d = 0; d < depth_hypos; ++d)
				{
					const float& cost = costs_filtered[d].at<float>(y, x);
					if (cost < min_cost)
					{
						min_cost = cost;
						min_idx = d;
					}
				}

				dm_tmp.at<float>(y, x) = (float)(min_idx);
			}
		}

		// compute sub-pixel depthmap 
		float f_d = 0.0f, f_d_plus = 0.0f, 
			f_d_minus = 0.0f, sub_depth = 0.0f;
		for (int y = 0; y < HEIGHT; ++y)
		{
			for (int x = 0; x < WIDTH; ++x)
			{
				const int& depth_int = (int)dm_tmp.at<float>(y, x);
				if (depth_int > depth_hypos - 2 || depth_int < 1)
				{
					depth_map.at<float>(y, x) = float(depth_int);
				}
				else
				{
					f_d = costs_filtered[depth_int].at<float>(y, x);
					f_d_plus = costs_filtered[depth_int + 1].at<float>(y, x);
					f_d_minus = costs_filtered[depth_int - 1].at<float>(y, x);
					sub_depth = float(depth_int) - (f_d_plus - f_d_minus) / (2.0f * (f_d_plus + f_d_minus - 2.0f * f_d));

					depth_map.at<float>(y, x) = sub_depth;
				}
			}
		}

		delete[] costs_filtered;
		costs_filtered = nullptr;

		// --------------------------- output
		cv::Mat mat2show;
		depth_map.copyTo(mat2show);
		mat2show.convertTo(mat2show, CV_8UC1);

		char buff[100];
		sprintf(buff, "./lamp_iter_%d.jpg", iter_i + 1);
		
		cv::imwrite(buff, mat2show);
		printf("=> iter %d done.\n\n", iter_i + 1);
	}
}

//void TestContours()
//{
//	// src image
//	Mat src = imread("C:/src.jpg", cv::IMREAD_COLOR);
//	if (src.empty())
//	{
//		cout << "[Err]: empty image.\n";
//	}
//
//	// dst image
//	Mat dst = Mat::zeros(src.rows, src.cols, CV_8UC3);
//}
//
//void TestSobelGuideFilter()
//{
//
//}

void TestLaplacianGuideFilter()
{
	// src image
	Mat src = imread("C:/src.jpg", cv::IMREAD_GRAYSCALE);

	// using laplacian edge as guide
	Mat I, blured;
	blur(src, blured, Size(5, 5));
	Laplacian(blured, I, -1, 5);

	// 二值图取反
	//bitwise_not(I, I);

	// input
	Mat p = imread("C:/MyColMap/colmap-dev/workspace/resultPro/depth_maps/IMG_2350.JPG.geometric.bin.jpg",
		cv::IMREAD_GRAYSCALE);

	if (I.empty() || p.empty())
	{
		cout << "[Err]: empty image." << endl;
		return;
	}

	int r = 5;
	double eps = 1e-6;

	eps *= 255 * 255;  // Because the intensity range of our images is [0, 255]

	// ---------------- run filter
	cv::Mat q = guidedFilter(I, p, r, eps);
	// ----------------

	// resize to show
	cv::Mat I_, p_, q_;
	cv::resize(I, I_,
		cv::Size(uint(p.cols * 0.3), uint(p.rows * 0.3)),
		cv::INTER_CUBIC);
	cv::resize(p, p_,
		cv::Size(uint(p.cols * 0.3), uint(p.rows * 0.3)),
		cv::INTER_CUBIC);
	cv::resize(q, q_,
		cv::Size(uint(p.cols * 0.3), uint(p.rows * 0.3)),
		cv::INTER_CUBIC);

	cv::imshow("laplacian guide", I_);
	cv::imshow("input", p_);
	cv::imshow("result", q_);
	cv::waitKey();

	imwrite("result.png", q);
}

void TestCannyAsGuideFilter()
{
	// src image
	Mat src = imread("C:/src.jpg", cv::IMREAD_GRAYSCALE);

	// canny edge
	Mat I, blured; // canny as guide
	blur(src, blured, Size(5, 5));
	Canny(blured, I, 0, 100, 5);

	// input
	Mat p = imread("C:/MyColMap/colmap-dev/workspace/resultPro/depth_maps/IMG_2350.JPG.geometric.bin.jpg",
		cv::IMREAD_GRAYSCALE);

	if (I.empty() || p.empty())
	{
		cout << "[Err]: empty image." << endl;
		return;
	}

	int r = 30;
	double eps = 1e-6;

	eps *= 255 * 255;  // Because the intensity range of our images is [0, 255]

	// ---------------- run filter
	cv::Mat q = guidedFilter(I, p, r, eps);
	// ----------------

	// resize to show
	cv::Mat I_, p_, q_;
	cv::resize(I, I_,
		cv::Size(uint(p.cols * 0.3), uint(p.rows * 0.3)),
		cv::INTER_CUBIC);
	cv::resize(p, p_,
		cv::Size(uint(p.cols * 0.3), uint(p.rows * 0.3)),
		cv::INTER_CUBIC);
	cv::resize(q, q_,
		cv::Size(uint(p.cols * 0.3), uint(p.rows * 0.3)),
		cv::INTER_CUBIC);

	cv::imshow("guide", I_);
	cv::imshow("input", p_);
	cv::imshow("result", q_);
	cv::waitKey();

	imwrite("result.png", q);
}

void TestSrcAsGuideFilter()
{
	//cv::Mat p = cv::imread("./src.png", CV_LOAD_IMAGE_GRAYSCALE);  // src
	//cv::Mat I = cv::imread("./guide.png", CV_LOAD_IMAGE_GRAYSCALE);  // guide

	// guide
	//Mat I = imread("C:/src.jpg", cv::IMREAD_COLOR);
	Mat I = imread("C:/src.jpg", cv::IMREAD_GRAYSCALE);

	//for (uint row = 0; row < I.rows; ++row)
	//{
	//	for (uint col = 0; col < I.cols; ++col)
	//	{
	//		const uchar data = I.at<uchar>(row, col);
	//		I.at<uchar>(row, col) = 255 - data;
	//	}
	//}

	// src
	Mat p = imread("C:/MyColMap/colmap-dev/workspace/resultPro/depth_maps/IMG_2350.JPG.geometric.bin.jpg",
		cv::IMREAD_GRAYSCALE);

	if (I.empty() || p.empty())
	{
		cout << "[Err]: empty image." << endl;
		return;
	}

	int r = 20;
	double eps = 1e-6;

	eps *= 255 * 255;  // Because the intensity range of our images is [0, 255]

	// ---------------- run filter
	cv::Mat q = guidedFilter(I, p, r, eps);
	// ----------------

	// 进一步滤波: 滤除从原图中引入的虚假深度
	for (int row = 0; row < q.rows; ++row)
	{
		for (int col = 0; col < q.cols; ++col)
		{
			int val = int(q.at<uchar>(row, col));
			if (val < 10)
			{
				//if (val != 0)
				//	cout << val << endl;
				if (val == 0)
					continue;
				q.at<uchar>(row, col) = uchar(0);
			}
		}
	}

	// resize to show
	cv::Mat I_, p_, q_;
	cv::resize(I, I_,
		cv::Size(uint(p.cols * 0.3), uint(p.rows * 0.3)),
		cv::INTER_CUBIC);
	cv::resize(p, p_,
		cv::Size(uint(p.cols * 0.3), uint(p.rows * 0.3)),
		cv::INTER_CUBIC);
	cv::resize(q, q_,
		cv::Size(uint(p.cols * 0.3), uint(p.rows * 0.3)),
		cv::INTER_CUBIC);

	char str[100];
	sprintf(str, "./SrcAsGuide_r%d.jpg", r);
	imwrite(str, q);

	cv::imshow("guide", I_);
	cv::imshow("input", p_);
	cv::imshow(str, q_);
	cv::waitKey();
}

void TestMyGuideFilter()
{
	// guide
	Mat I = imread("C:/MyColMap/colmap-dev/workspace/SrcMVS/images/IMG_2339.JPG",
		cv::IMREAD_COLOR);  

	// src
	Mat p = imread("C:/MyColMap/colmap-dev/workspace/resultPro/depth_maps/IMG_2339.JPG.geometric.bin.jpg",
		cv::IMREAD_GRAYSCALE);  

	if (I.empty() || p.empty())
	{
		cout << "[Err]: empty image." << endl;
		return;
	}
	cout << "Depth map channels: " << I.channels() << endl;

	// 对guide下采样
	Mat I_;
	resize(I, I_,
		Size(p.cols, p.rows),
		INTER_CUBIC);

	int r = 6;
	uint win_size = 2 * r + 1;
	double eps = 1e-2;

	//eps *= 255 * 255;  // Because the intensity range of our images is [0, 255]

	cv::Mat q = guidedFilter(I_, p, r, eps);

	char path_[100];
	sprintf(path_, "c:/guide_filter_new_%d×%d.jpg", win_size, win_size);
	imwrite(path_, q);

	cv::resize(q, q,
		Size(uint(q.cols * 0.4f), uint(q.rows * 0.4f)),
		cv::INTER_CUBIC);

	cv::imshow(path_, q);
	cv::waitKey();
}

void TestCanny()
{
	const string src_path = string("c:/guide.jpg");

	Mat src = imread(src_path, IMREAD_GRAYSCALE);
	if (src.empty())
	{
		cout << "[Err]: empty src.\n";
		return;
	}

	Mat blured, canny_edge, canny_edge_;
	blur(src, blured, Size(3, 3));

	Canny(blured, canny_edge, 100, 200, 3);

	resize(canny_edge,
		canny_edge_, 
		Size(uint(src.cols * 0.3f), uint(src.rows * 0.3f)), 
		INTER_CUBIC);

	imshow("canny", canny_edge_);
	waitKey();

	imwrite("c:/canny_edge.jpg", canny_edge);
}

void TestContourGuideFilter()
{
	// guide
	Mat I = imread("C:/src.jpg", cv::IMREAD_GRAYSCALE);

	// src
	Mat p = imread("C:/MyColMap/colmap-dev/workspace/resultPro/depth_maps/IMG_2350.JPG.geometric.bin.jpg",
		cv::IMREAD_GRAYSCALE);

	if (I.empty() || p.empty())
	{
		cout << "[Err]: empty image." << endl;
		return;
	}
	cout << "Depth map channels: " << I.channels() << endl;

	Mat blured, canny_edge, laplace_edge;
	blur(I, blured, Size(3, 3));

	Canny(blured, canny_edge, 100, 200, 3);
	//Laplacian(blured, laplace_edge, -1);

	assert(I.size() == p.size());

	Mat dst(p.rows, p.cols, p.type());
	int k = 3;

	// ------------ 用原图(RGB or gray?)引导深度图
	uint win_size = 2 * k - 1;
	cout << "Filter window size: " << win_size << "×" << win_size << endl;

	dst = GuideFilter(p, canny_edge, win_size, 1000.0f);
	//dst = GuideFilter(p, laplace_edge, win_size, 1e-2);

	// ------------ 联合双边滤波
	//ximgproc::jointBilateralFilter(I, p, dst, 2 * k - 1, 0.05*sqrtf(3), k - 1);

	// 转换引导后的图像到原图类型
	dst.convertTo(dst, CV_8UC1);

	if (!dst.empty())
	{
		std::string();
		char guide_filter_path[100];
		sprintf(guide_filter_path, "c:/guide_filter_%d×%d.jpg", win_size, win_size);
		imwrite(guide_filter_path, dst);

		Mat p_, dst_, edge_;
		resize(p, p_,
			Size(uint(p.cols * 0.4f), uint(p.rows * 0.4f)),
			INTER_CUBIC);
		p_.convertTo(p_, CV_8UC1);

		resize(dst, dst_,
			Size(uint(p.cols * 0.4f), uint(p.rows * 0.4f)),
			INTER_CUBIC);
		resize(canny_edge, edge_,
			Size(uint(p.cols * 0.4f), uint(p.rows * 0.4f)),
			INTER_CUBIC);
		//resize(laplace_edge, edge_,
		//	Size(uint(p.cols * 0.4f), uint(p.rows * 0.4f)),
		//	INTER_CUBIC);

		imshow("p", p_);
		imshow("canny gudie", edge_);
		//imshow("laplacian gudie", edge_);

		imshow(guide_filter_path, dst_);
		waitKey();
	}

}

void TestContourGuideFilterNew()
{
	// guide
	Mat I = imread("C:/guide.jpg", cv::IMREAD_GRAYSCALE);

	// src
	Mat p = imread("C:/MyColMap/colmap-dev/workspace/resultPro/depth_maps/IMG_2339.JPG.geometric.bin.jpg",
		cv::IMREAD_GRAYSCALE);

	if (I.empty() || p.empty())
	{
		cout << "[Err]: empty image." << endl;
		return;
	}
	cout << "Depth map channels: " << I.channels() << endl;

	Mat blured, canny_edge, canny_edge_;
	blur(I, blured, Size(3, 3));

	Canny(blured, canny_edge, 100, 200, 3);

	int r = 1;
	uint win_size = 2 * r + 1;
	double eps = 1e-6;

	//eps *= 255 * 255;  // Because the intensity range of our images is [0, 255]

	Mat q = guidedFilter(canny_edge, p, r, eps);

	char path_[100];
	sprintf(path_, "c:/guide_filter_new_%d×%d.jpg", win_size, win_size);
	imwrite(path_, q);

	Mat guide;
	resize(canny_edge, guide,
		Size(uint(I.cols * 0.4f), uint(I.rows *  0.4f)),
		INTER_CUBIC);

	resize(q, q,
		Size(uint(q.cols * 0.4f), uint(q.rows * 0.4f)),
		cv::INTER_CUBIC);

	imshow("guide", guide);
	imshow(path_, q);
	waitKey();
}

//int main()
//{
//	//TestGuideFilter(0);
//
//	//TestPatchMatchOptimize();
//
//	TestDepthSR();
//	//TestMatVector();
//
//	//TestSrcAsGuideFilter();
//	//TestCannyAsGuideFilter();
//	//TestLaplacianGuideFilter();
//
//	//TestMyGuideFilter();
//
//	//TestContourGuideFilterNew();
//	//TestContourGuideFilter();
//
//	//TestCanny();
//
//	cout << "Test done." << endl;
//
//	system("pause");
//	return 0;
//}

/*
// split to 3 channels and guide filter each channel
// then merge
//vector<Mat> src_split(3), result_img(3);
//split(src, src_split);

//for (uint i = 0; i < 3; ++i)
//{
//	Mat tmp;
//	src_split[i].convertTo(tmp, CV_64FC1, 1.0 / 255.0);

//	Mat img_clone = tmp.clone();
//	Mat result = GuideFilter(tmp, guide_, 9, 0.01);  // 3通道, 同一个guide
//	result_img.push_back(result);
//}
//merge(result_img, dst);
*/