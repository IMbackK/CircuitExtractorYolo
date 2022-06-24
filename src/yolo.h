#pragma once
#include <cstdint>
#include <opencv2/core/types.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <string>
#include <vector>

class Yolo5
{
public:
	static constexpr int TRAIN_SIZE_X = 1024;
	static constexpr int TRAIN_SIZE_Y = 1024;
	static constexpr double DETECTION_THRESH = 0.5;

	struct DetectedClass
	{
		size_t classId;
		float prob;
		cv::Rect rect;
	};

private:
	cv::dnn::Net net;

public:
	Yolo5(const cv::dnn::Net &netI);
	Yolo5(const std::string& fileName);
	std::vector<DetectedClass> detect(const cv::Mat& image);
};


