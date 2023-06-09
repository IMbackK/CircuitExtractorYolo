#pragma once
#include <cstdint>
#include <opencv2/core/types.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <string>
#include <vector>

class Yolo5
{
public:
	static constexpr double DETECTION_THRESH = 0.15;
	static constexpr double NMS_THRESH = 0.3;
	static constexpr double SCORE_THRES = 0.3;
	static constexpr int YOLO_N_VECTOR_DEPTH = 25200;

	struct DetectedClass
	{
		size_t classId;
		float prob;
		cv::Rect rect;
	};

private:
	size_t numClasses;
	cv::dnn::Net net;
	int dimensions;
	const int trainSizeX;
	const int trainSizeY;

private:
	cv::Mat resizeWithBorder(const cv::Mat& mat);
	cv::Mat prepare(const cv::Mat& mat);
	void transformCord(std::vector<DetectedClass>& detections, const cv::Size& matSize);

public:
	Yolo5(const cv::dnn::Net &netI, size_t numClassesI, int trainSizeXIn = 640, int trainSizeYIn = 640);
	Yolo5(const std::string& fileName, size_t numClassesI, int trainSizeXIn = 640, int trainSizeYIn = 640);
	Yolo5(size_t networkDataSize, const char* networkData, size_t numCassesI, int trainSizeXIn = 640, int trainSizeYIn = 640);
	std::vector<DetectedClass> detect(const cv::Mat& image);

	static void drawDetection(cv::Mat& image, const DetectedClass& detection);
};


