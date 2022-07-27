#pragma once

#include "element.h"
#include <cstddef>
#include <cstdint>

class Net
{
private:
	std::vector<size_t> connectedEndpointIndecies;
	uint64_t id;

public:
	std::vector<cv::Point2i> endpoints;
	std::vector<cv::Point2i> nodes;
	std::vector<cv::Vec4f> lines;
	std::vector<Element*> elements;

private:
	bool pointIsFree(const cv::Point2i& point, const size_t ignore, double tollerance);

public:
	Net();
	Net(cv::Point a, cv::Point b);
	bool operator==(const Net& net) const;
	void draw(cv::Mat& image, const cv::Scalar* color = nullptr) const;
	void computePoints(double tollerance = 10);
	void coordScale(double factor);
	bool addElement(Element* element, DirectionHint hint = C_DIRECTION_UNKOWN, double tolleranceFactor = 1.0);
	cv::Rect endpointRect() const;
	cv::Point center() const;
	uint64_t getId() const;
	size_t elementCount() const;
};
