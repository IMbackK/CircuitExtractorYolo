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
	E_TYPE_COMPOSIT,
	E_TYPE_UNKOWN,
	E_TYPE_COUNT,
} ElementType;

typedef enum {
	C_DIRECTION_HORIZ,
	C_DIRECTION_VERT,
	C_DIRECTION_UNKOWN
} DirectionHint;

class Element
{
private:

public:
	ElementType type;
	cv::Rect rect;
	cv::Mat image;
	float prob;

public:
	std::string getString() const;
};

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
	void draw(cv::Mat& image, const cv::Scalar* color = nullptr) const;
	void computePoints(double tollerance = 10);
	void coordScale(double factor);
	bool addElement(Element* element, DirectionHint hint = C_DIRECTION_UNKOWN, double tolleranceFactor = 1.0);
	cv::Rect endpointRect() const;
	cv::Point center() const;
};

class Circut
{
private:
	std::string model;
public:

	float prob;
	cv::Mat image;
	std::vector<Element> elements;
	std::vector<Net> nets;

private:
	static bool moveConnectedLinesIntoNet(Net& net, size_t index, std::vector<cv::Vec4f>& lines, double tollerance);
	static std::vector<Net> sortLinesIntoNets(std::vector<cv::Vec4f> lines, double tollerance);
	static void balanceBrackets(std::string& str);

	void removeUnconnectedNets();
	size_t getStartingIndex(DirectionHint hint) const;
	int64_t getOpositNetIndex(const Element* element, Net* net) const;
	size_t getEndingIndex(DirectionHint hint) const;
	void getStringForPath(std::string& str, const Element* element,
	                      std::vector<const Element*>& handled, size_t netIndex,
	                      size_t endNetIndex, size_t startNetIndex);

	std::vector<Net*> getElementAdjacentNets(const Element* const element);

public:
	cv::Mat ciructImage() const;
	void detectElements(Yolo5* yolo);
	void detectNets(DirectionHint hint = C_DIRECTION_UNKOWN);
	std::string getString(DirectionHint hint = C_DIRECTION_UNKOWN);
};
