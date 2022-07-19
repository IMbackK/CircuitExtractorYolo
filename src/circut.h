#pragma once

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

#include "yolo.h"

typedef enum {
	E_TYPE_R = 0,
	E_TYPE_C,
	E_TYPE_L,
	E_TYPE_P,
	E_TYPE_W,
	E_TYPE_SOURCE,
	E_TYPE_UNKOWN,
	E_TYPE_COUNT,
} ElementType;

struct Element
{
	ElementType type;
	cv::Rect rect;
	cv::Mat image;
	float prob;
};

class Net
{
public:
	std::vector<cv::Point2i> endpoints;
	std::vector<cv::Point2i> nodes;
	std::vector<cv::Vec4f> lines;

private:
	bool pointIsFree(const cv::Point2i& point, const size_t ignore, double tollerance);

public:

	void draw(cv::Mat& image) const;
	void computePoints(double tollerance = 10);
	void coordScale(double factor);
};

class Circut
{
public:

	std::string model;
	float prob;
	cv::Mat image;
	std::vector<Element> elements;
	std::vector<Net> nets;

private:
	static bool moveConnectedLinesIntoNet(Net& net, size_t index, std::vector<cv::Vec4f>& lines, double tollerance);
	static std::vector<Net> sortLinesIntoNets(std::vector<cv::Vec4f> lines, double tollerance);

public:
	cv::Mat ciructImage() const;

	void detectElements(Yolo5* yolo);
	void detectNets();
};
