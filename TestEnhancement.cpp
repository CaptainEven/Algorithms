#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include<fstream>
//#include<algorithm>
#include<math.h>

#include<opencv2/opencv.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/ximgproc.hpp>


template<class T>
inline T Clamp(const T& x, const T& min, const T& max)
{
	if (x > max)
		return max;
	if (x < min)
		return min;
	return x;
}

// 增强一个通道
int EnhanceOneChannel(cv::Mat& img)
{
	if (img.empty())
	{
		printf("[Err]: empty image");
		return -1;
	}

	double GlobalSum = 0.0;
	double GlobalMean = 0.0;
	double GlobalSum2 = 0.0;
	double GlobalVar = 0.0;

	// LineBytes: 原图一行所占字节数
	//printf("LineBytes: %d\n", (int)img.step[0]);
	const size_t& LineBytes = img.step[0];

	for (size_t y = 0; y < img.rows; ++y)
	{
		for (size_t x = 0; x < img.cols; ++x)
		{
			size_t index = y * LineBytes + x;
			GlobalSum += (double)img.data[index];
			GlobalSum2 += (double)img.data[index] * (double)img.data[index];
		}
	}

	double TotalNumPixs = (double(img.rows) * double(img.cols));
	GlobalMean = GlobalSum / TotalNumPixs;
	GlobalVar = (GlobalSum2 - (GlobalSum * GlobalSum) / TotalNumPixs) / TotalNumPixs;

	const int Size = 7;  // Window size
	const int WinNumPixs = Size * Size;  // windows pixel number
	const int Pad = (Size - 1) / 2;  // Padding

	// 填充padding
	cv::Mat img_pad;
	cv::copyMakeBorder(img, img_pad, Pad, Pad, Pad, Pad, cv::BORDER_REFLECT);

	// 遍历img_pad
	// pad图一行所占字节数
	const size_t& PadLineBytes = img_pad.step[0];
	for (size_t y = Pad; y < img_pad.rows - Pad; ++y)
	{
		for (size_t x = Pad; x < img_pad.cols; ++x)
		{
			double Sum = 0.0;
			double Mean = 0.0;
			double Sum2 = 0.0;
			double Var = 0.0;
			size_t index_pad = y * PadLineBytes + x;
			size_t index = (y - Pad) * LineBytes + (x - Pad);

			// 取窗口
			for (int r = -Pad; r <= Pad; ++r)
			{
				for (int c = -Pad; c <= Pad; ++c)
				{
					Sum += img_pad.data[index_pad + r * PadLineBytes + c];
					Sum2 += img_pad.data[index_pad + r * PadLineBytes + c] \
						* img_pad.data[index_pad + r * PadLineBytes + c];
				}
			}

			Mean = Sum / double(WinNumPixs);
			Var = (Sum2 - (Sum * Sum) / WinNumPixs) / WinNumPixs;

			double CG = GlobalVar / Var;
			//double CG = Var / GlobalVar;
			//CG += 1.0;

			CG = CG <= 2.0 ? CG : 2.0;
			//if (CG != 2.0)
			//	printf("CG: %.3f\n", (float)CG);

			double PixData = Mean + CG * ((double)img_pad.data[index_pad] - Mean);
			PixData = Clamp(PixData, 0.0, 255.0);
			img.data[index] = (uchar)PixData;
		}
	}

	return 0;
}

inline int Enhance(cv::Mat& img)
{
	std::vector<cv::Mat> channels;  // b,r,g
	cv::split(img, channels);

	int ret0 = EnhanceOneChannel(channels[0]);
	int ret1 = EnhanceOneChannel(channels[1]);
	int ret2 = EnhanceOneChannel(channels[2]);

	cv::merge(channels, img);

	if (ret0 == -1 || ret1 == -1 || ret2 == -1)
		return -1;
	return 0;
}

void Test3()
{
	std::string img_path = std::string("./Courtyard_SRC2.jpg");
	cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);
	
	Enhance(img);

	cv::imwrite("./Courtyard_ENH2.jpg", img);
}

void Test2()
{
	std::string img_path = std::string("./road.jpg");
	cv::Mat img = cv::imread(img_path, cv::IMREAD_GRAYSCALE);

	EnhanceOneChannel(img);
	
	cv::imwrite("./road_enh_pad.jpg", img);
}

int Test1()
{
	std::string img_path = std::string("./road.jpg");

	cv::Mat img = cv::imread(img_path, cv::IMREAD_GRAYSCALE);

	if (img.empty())
	{
		printf("[Err]: empty image");
		return -1;
	}

	double GlobalSum = 0.0;
	double GlobalMean = 0.0;
	double GlobalSum2 = 0.0;
	double GlobalVar = 0.0;

	// LineBytes:一行所占字节数
	printf("LineBytes: %d\n", (int)img.step[0]);  
	const size_t& LineBytes = img.step[0];

	for (size_t y = 0; y < img.rows; ++y)
	{
		for (size_t x = 0; x < img.cols; ++x)
		{
			size_t index = y * LineBytes + x;
			GlobalSum += (double)img.data[index];
			GlobalSum2 += (double)img.data[index] * (double)img.data[index];
		}
	}

	double TotalNumPixs = (double(img.rows) * double(img.cols));
	GlobalMean = GlobalSum / TotalNumPixs;
	GlobalVar = (GlobalSum2 - (GlobalSum * GlobalSum) / TotalNumPixs) / TotalNumPixs;

	const int Size = 7;  // Window size
	const int WinNumPixs = Size * Size;  // windows pixel number
	const int Pad = (Size - 1) / 2;  // Padding

	// 填充padding...
	cv::Mat img_pad;
	cv::copyMakeBorder(img, img_pad, Pad, Pad, Pad, Pad, cv::BORDER_REFLECT);

	//cv::Mat img_pad_rs;
	//cv::resize(img_pad, img_pad_rs, cv::Size(img_pad.cols * 0.15, img_pad.rows * 0.15));
	//cv::imshow("pad", img_pad_rs);
	//cv::waitKey();

	//// 遍历原图每一个像素
	//for (size_t y = Pad; y < img.rows - Pad; ++y)
	//{
	//	for (size_t x = Pad; x < img.cols - Pad; ++x)
	//	{
	//		double Sum = 0.0;
	//		double Mean = 0.0;
	//		double Sum2 = 0.0;
	//		double Var = 0.0;
	//		size_t index = y * LineBytes + x;

	//		// 取窗口
	//		for (int r = -Pad; r <= Pad; ++r)
	//		{
	//			for (int c = -Pad; c <= Pad; ++c)
	//			{
	//				Sum += img.data[index + r * LineBytes + c];
	//				Sum2 += img.data[index + r * LineBytes + c] \
	//					* img.data[index + r * LineBytes + c];
	//			}
	//		}

	//		Mean = Sum / double(WinNumPixs);
	//		Var = (Sum2 - (Sum * Sum) / WinNumPixs) / WinNumPixs;

	//		double CG = GlobalVar / Var;
	//		//double CG = Var / GlobalVar;
	//		//CG += 1.0;

	//		CG = CG <= 2.0 ? CG : 2.0;
	//		//if (CG != 2.0)
	//		//	printf("CG: %.3f\n", (float)CG);

	//		double PixData = Mean + CG * ((double)img.data[index] - Mean);
	//		PixData = Clamp(PixData, 0.0, 255.0);
	//		img.data[index] = (uchar)PixData;
	//	}
	//}

	// 遍历img_pad
	const size_t& PadLineBytes = img_pad.step[0];
	for (size_t y = Pad; y < img_pad.rows - Pad; ++y)
	{
		for (size_t x = Pad; x < img_pad.cols; ++x)
		{
			double Sum = 0.0;
			double Mean = 0.0;
			double Sum2 = 0.0;
			double Var = 0.0;
			size_t index_pad = y * PadLineBytes + x;
			size_t index = (y - Pad) * LineBytes + (x - Pad);

			// 取窗口
			for (int r = -Pad; r <= Pad; ++r)
			{
				for (int c = -Pad; c <= Pad; ++c)
				{
					Sum += img_pad.data[index_pad + r * PadLineBytes + c];
					Sum2 += img_pad.data[index_pad + r * PadLineBytes + c] \
						* img_pad.data[index_pad + r * PadLineBytes + c];
				}
			}

			Mean = Sum / double(WinNumPixs);
			Var = (Sum2 - (Sum * Sum) / WinNumPixs) / WinNumPixs;

			double CG = GlobalVar / Var;
			//double CG = Var / GlobalVar;
			//CG += 1.0;

			CG = CG <= 2.0 ? CG : 2.0;
			//if (CG != 2.0)
			//	printf("CG: %.3f\n", (float)CG);

			double PixData = Mean + CG * ((double)img_pad.data[index_pad] - Mean);
			PixData = Clamp(PixData, 0.0, 255.0);
			img.data[index] = (uchar)PixData;
		}
	}

	cv::imwrite("./road_enh_pad.jpg", img);
		
	return 0;
}


int main()
{
	Test3();
	printf("Done\n");

	system("pause");
	return 0;
}