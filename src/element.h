#pragma once

#include <opencv2/core/mat.hpp>
#include <string>

#include "utils.h"

typedef enum {
	E_TYPE_R = 0,
	E_TYPE_C,
	E_TYPE_L,
	E_TYPE_P,
	E_TYPE_W,
	E_TYPE_SOURCE,
	E_TYPE_COMPOSIT,
	E_TYPE_UNKOWN,
	E_TYPE_COUNT,
} ElementType;

class Element
{
private:
	ElementType type;
	cv::Rect rect;
	float prob;

public:
	cv::Mat image;

public:
	Element(ElementType type, cv::Rect rect, float prob = 1);
	ElementType getType() const;
	cv::Rect getRect() const;
	double getProb() const;
	std::string getString() const;
};
