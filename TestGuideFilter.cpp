#include <iostream>   

#include <opencv2/core/core.hpp>                    
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>   

using namespace std;
using namespace cv;

Mat guidedFilter(Mat& src, Mat& guided, int radius, double eps);  // 引导滤波器

Mat guidedFilter(Mat& src, Mat& guided, int radius, double eps)
{
	//------------------------    
	src.convertTo(src, CV_64FC1);
	guided.convertTo(guided, CV_64FC1);

	Mat mean_p, mean_I, mean_Ip, mean_II;
	boxFilter(src, mean_p, CV_64FC1, Size(radius, radius));
	boxFilter(guided, mean_I, CV_64FC1, Size(radius, radius));
	boxFilter(src.mul(guided), mean_Ip, CV_64FC1, Size(radius, radius));
	boxFilter(guided.mul(guided), mean_II, CV_64FC1, Size(radius, radius));

	//------------------------    
	Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);
	Mat var_I = mean_II - mean_I.mul(mean_I);

	//------------------------
	Mat a = cov_Ip / (var_I + eps);
	Mat b = mean_p - a.mul(mean_I);

	Mat mean_a, mean_b;
	boxFilter(a, mean_a, CV_64FC1, Size(radius, radius));
	boxFilter(b, mean_b, CV_64FC1, Size(radius, radius));

	Mat dstImage = mean_a.mul(src) + mean_b;
	return dstImage;
}

int TestGuideFilter()
{
	//--------------
	Mat dst;  //最后结果图像  
	vector<Mat> vSrc(3), vResult(3);

	//--------------
	Mat src = imread("c:/guide.jpg");
	if (!src.data)
	{
		cout << "Read image error!\n";
		system("pause");
		return -1;
	}

	resize(src, src,
		Size(src.cols * 0.3, src.rows * 0.3),
		INTER_CUBIC);

	imshow("src", src);
	//waitKey();

	//--------------
	split(src, vSrc);

	// BRG channels
	for (int i = 0; i < 3; ++i)
	{
		Mat tmp;
		vSrc[i].convertTo(tmp, CV_64FC1, 1.0 / 255.0);

		Mat tmp_clone = tmp.clone();
		Mat result_channel = guidedFilter(tmp, tmp_clone, 5, 0.01);
		vResult[i] = result_channel;
	}

	merge(vResult, dst);

	imshow("guide", dst);
	waitKey(0);
}

int TestSplitMerge()
{
	Mat src = imread("c:/guide.jpg");
	if (!src.data)
	{
		cout << "[Err]: empty img." << endl;
		return -1;
	}
	resize(src,
		   src,
		   Size(uint(src.cols * 0.3), uint(src.rows * 0.3)),
		   INTER_CUBIC);

	imshow("src", src);
	waitKey();

	// ------------------- 
	vector<Mat> channels;

	split(src, channels);

	Mat blue_channel = channels[0];
	Mat green_channel = channels[1];
	Mat red_channel = channels[2];

	// ------------------- 
	vector<Mat> mbgr(3);
	Mat bk_1(src.size(), CV_8UC1, Scalar(0));

	Mat img_B(src.size(), CV_8UC3);
	mbgr[0] = channels[0];
	mbgr[1] = bk_1;
	mbgr[2] = bk_1;

	merge(mbgr, img_B);

	imshow("img B", img_B);
	waitKey();

	return 0;
}

//int main()
//{
//	TestGuideFilter();
//
//	//TestSplitMerge();
//
//	system("pause");
//	return 0;
//}
