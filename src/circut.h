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
