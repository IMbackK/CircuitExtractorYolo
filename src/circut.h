#pragma once

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <string>

#include "yolo.h"
#include "element.h"
#include "net.h"

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
	std::vector<Net*> getElementAdjacentNets(const Element* const element);
	size_t appendStringForSerisPath(std::string& str, const Element* element, std::vector<const Element*>& handled, size_t netIndex, size_t endNetIndex, size_t startNetIndex);
	void appendStringForParalellPath(std::string& str, const Element* element, std::vector<const Element*>& handled, size_t netIndex, size_t endNetIndex, size_t startNetIndex);
	Element* findUaccountedPathStartingElement(DirectionHint hint, size_t start, size_t stop, std::vector<const Element*>& handled);

public:
	cv::Mat ciructImage() const;
	void detectElements(Yolo5* yolo);
	void detectNets(DirectionHint hint = C_DIRECTION_UNKOWN);
	std::string getString(DirectionHint hint = C_DIRECTION_UNKOWN);
};
