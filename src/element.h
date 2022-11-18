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
	E_TYPE_NODE,
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
	std::string string;
	cv::Mat image;

private:
	int getPrecedence() const;
	static int getPrecedence(ElementType type);
	static ElementType typeFromChar(char c);

public:
	Element(ElementType type = E_TYPE_UNKOWN, cv::Mat image = cv::Mat(), cv::Rect rect = cv::Rect(), float prob = 1);
	Element(const Element& a, const Element& b, bool serial = true);
	ElementType getType() const;
	cv::Rect getRect() const;
	cv::Point center() const;
	double getProb() const;
	std::string getString() const;
	const cv::Mat getImage() const;
};
