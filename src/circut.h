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
};

class Circut
{
public:

	std::string model;
	float prob;
	cv::Mat image;
	std::vector<Element> elements;

	cv::Mat ciructImage() const;

	void getElements(Yolo5* yolo);
};
