#ifndef COLMAP_SRC_MVS_WORKSPACE_H_
#define COLMAP_SRC_MVS_WORKSPACE_H_

#include "consistency_graph.h"
#include "depth_map.h"
#include "model.h"
#include "normal_map.h"

#include "SLICSuperpixels.h"


namespace colmap {
	namespace mvs {

		class Workspace {
		public:
			struct Options {
				// The maximum cache size in gigabytes.
				double cache_size = 32.0;

				// Maximum image size in either dimension.
				int max_image_size = 400;  // -1

				// Whether to read image as RGB or gray scale.
				bool image_as_rgb = true;

				////是否利用细节增强的图像
				bool bDetailEnhance = true;

				////是否利用结构增强的图像
				bool bStructureEnhance = false;

				////是否进行降采样处理(对图像尺度和摄像机参数做修改，稀疏三维模型不变)
				bool bDown_sampling = false;

				//降采样的尺度
				float fDown_scale = 4.0f;

				bool bOurs = false;  // 联合双边传播上采样
				bool bOursFast = false;  // 快速联合双边传播上采样方法

				bool bBilinearInterpolation = 0;  // 双线性插值上采样

				bool bFastBilateralSolver = 0;  // 快速双边求解器

				bool bJointBilateralUpsampling = 0;  // 联合双边上采样

				bool bBilateralGuidedUpsampling = 0;  // 双边引导上采样

				// Location and type of workspace.
				std::string workspace_path;
				std::string workspace_format;
				std::string input_type;
				std::string input_type_geom;
				std::string newPath;
				std::string src_img_dir;

				//去扭曲的目录
				std::string undistorte_path;

				// 超像素分割目录
				std::string slic_path;
			};

			// 构造函数: 从bundler数据中读取稀疏点云信息
			Workspace(const Options& options);

			const Model& GetModel() const;
			const cv::Mat& GetBitmap(const int image_id);
			const DepthMap& GetDepthMap(const int image_id) const;
			const NormalMap& GetNormalMap(const int image_id) const;
			const ConsistencyGraph& GetConsistencyGraph(const int image_id) const;

			const void ReadDepthAndNormalMaps(const bool isGeometric);

			const std::vector<DepthMap>& GetAllDepthMaps() const;
			const std::vector<NormalMap>& GetAllNormalMaps() const;

			//将算出来的结果，写到workspace对应的变量里面
			void WriteDepthMap(const int image_id, const DepthMap &depthmap);
			void WriteNormalMap(const int image_id, const NormalMap &normalmap);
			void WriteConsistencyGraph(const int image_id, const ConsistencyGraph &consistencyGraph);

			//执行超像素分割
			void runSLIC(const std::string &path);

			//将三维点投影到图像中
			void showImgPointToSlicImage(const std::string &path);

			//对深度图和法向图进行上采样，同时修改model中的图像信息
			void UpSampleMapAndModel();

			// Get paths to bitmap, depth map, normal map and consistency graph.
			std::string GetBitmapPath(const int image_id) const;
			std::string GetDepthMapPath(const int image_id, const bool isGeom) const;
			std::string GetNormalMapPath(const int image_id, const bool isGeom) const;
			std::string GetConsistencyGaphPath(const int image_id) const;

			// Return whether bitmap, depth map, normal map, and consistency graph exist.
			bool HasBitmap(const int image_id) const;
			bool HasDepthMap(const int image_id, const bool isGeom) const;
			bool HasNormalMap(const int image_id, const bool isGeom) const;

			float GetDepthRange(const int image_id, bool isMax) const;

			void jointBilateralUpsampling(const cv::Mat &joint, const cv::Mat &lowin, const float upscale,
				const double sigma_color, const double sigma_space, int radius, cv::Mat &highout) const;

			void jointBilateralPropagationUpsampling(const cv::Mat &joint, const cv::Mat &lowDepthMat, const cv::Mat &lowNormalMat, const float *refK,
				const float upscale, const double sigma_color, const double sigma_space, const int radius, cv::Mat &highDepthMat) const;

			void jointBilateralDepthMapFilter1(const cv::Mat &srcDepthMap, const cv::Mat &srcNormalMap, const cv::Mat &srcImage, const float *refK,
				const int radius, const double sigma_color, const double sigma_space, DepthMap &desDepMap, NormalMap &desNorMap, const bool DoNormal)const;

			float PropagateDepth(const float *refK, const float depth1, const float normal1[3],
				const float row1, const float col1, const float row2, const float col2) const;

			void SuitNormal(const int row, const int col, const float *refK, float normal[3]) const;

			//对法向量图进行类中值滤波
			void NormalMapMediaFilter(const cv::Mat &InNormalMapMat, cv::Mat &OutNormalMapMat, const int windowRadis)const;
			void NormalMapMediaFilter1(const cv::Mat &InNormalMapMat, cv::Mat &OutNormalMapMat, const int windowRadis)const;
			void NormalMapMediaFilterWithDepth(const cv::Mat &InNormalMapMat, cv::Mat &OutNormalMapMat,
				const cv::Mat &InDepthMapMat, cv::Mat &OutDepthMapMat, int windowRadis)const;

			std::vector<cv::Point3f> sparse_normals_;

			std::string GetFileName(const int image_id, const bool isGeom) const;

			void newPropagation(const cv::Mat &joint, const cv::Mat &lowDepthMat, const cv::Mat &lowNormalMat, const float *refK,
				const float upscale, const double sigma_color, const double sigma_space, int radius, const int maxSrcPoint,
				cv::Mat &highDepthMat, cv::Mat &highNormalMat) const;

			void newPropagationFast(const cv::Mat &joint, const cv::Mat &lowDepthMat, const cv::Mat &lowNormalMat, const float *refK,
				const double sigma_color, const double sigma_space, int radius, const int maxSrcPoint,
				cv::Mat &outDepthMat, cv::Mat &outNormalMat) const;

			// 测试为CV_32F数据进行斑点滤波操作
			template<typename T>
			void FilterSpeckles(cv::Mat & img, T newVal, int maxSpeckleSize, T maxDiff);

			// @even测试深度图
			void TestDepthmap();

			//将src和enhance的结果合并在一起
			void MergeDepthNormalMaps(const bool haveMerged = false, const bool selectiveJBPF = false);

			//对深度和法向量图进行有选择性的联合双边(传播)滤波插值
			void selJointBilateralPropagateFilter(const cv::Mat& joint, 
				const DepthMap& depthMap, const NormalMap& normalMap,
				const float *refK,
				const double sigma_color, const double sigma_space,
				int radius, const int maxSrcPoint,
				DepthMap& outDepthMap, NormalMap& outNormalMap) const;

			// 噪声感知滤波(双边滤波的变种)Noise-aware filter
			void NoiseAwareFilter(const cv::Mat& joint, 
				DepthMap& depthMap, const NormalMap& normalMap,
				const float* refK,
				const double& sigma_space, const double& sigma_color, const double& sigma_depth,
				const float& THRESH,
				const float& eps, const float& tau,
				const bool is_propagate,
				int radius, 
				DepthMap& outDepthMap, NormalMap& outNormalMap) const;

			//  joint-trilateral-upsampling: JTU
			void JTU(const cv::Mat& joint,
				DepthMap& depthMap, const NormalMap& normalMap,
				const float* refK,
				const double& sigma_space, const double& sigma_color, double& sigma_depth,
				const float& THRESH,
				const bool is_propagate,
				int radius,
				DepthMap& outDepthMap, NormalMap& outNormalMap) const;

			//对深度图和法向量图进行联合双边滤波
			void jointBilateralFilter_depth_normal_maps(const cv::Mat &joint, const DepthMap &depthMap, const NormalMap &normalMap,
				const float *refK, const double sigma_color, const double sigma_space, int radius,
				DepthMap &outDepthMap, NormalMap &outNormalMap) const;

			void distanceWeightFilter(const DepthMap &depthMap, const NormalMap &normalMap,
				const float *refK, const double sigma_color, const double sigma_space, int radius,
				DepthMap &outDepthMap, NormalMap &outNormalMap) const;

		private:

			Options options_;
			Model model_;
			bool hasReadMapsPhoto_;  // 是否读入图像一致性map图
			bool hasReadMapsGeom_;  // 是否读入几何一致性map图，两者不能同时为true；
			std::vector<bool> hasBitMaps_;
			std::vector<cv::Mat> bitMaps_;
			std::vector<DepthMap> m_depth_maps;
			std::vector<NormalMap> m_normal_maps;
			std::vector<std::pair<float, float>> depth_ranges_;
			std::vector<cv::Mat> slicLabels_;

		};

	}  // namespace mvs
}  // namespace colmap

#endif  // COLMAP_SRC_MVS_WORKSPACE_H_
