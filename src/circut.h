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
	std::vector<Element*> elements;
	std::vector<Net> nets;

private:
	static bool moveConnectedLinesIntoNet(Net& net, size_t index, std::vector<cv::Vec4f>& lines, double tollerance);
	static std::vector<Net> sortLinesIntoNets(std::vector<cv::Vec4f> lines, double tollerance);
	static Net* netFromId(std::vector<Net>& nets, uint64_t id);
	static bool colapseSerial(std::vector<Net>& nets, std::vector<Element*>& joinedElements, uint64_t startingId, uint64_t endingId);
	static bool colapseParallel(std::vector<Net>& nets, std::vector<Element*>& joinedElements);
	static uint64_t getOpositNetId(const Element* element, const Net& net, const std::vector<Net>& netsL);

	void removeUnconnectedNets();
	uint64_t getStartingNetId(DirectionHint hint) const;
	uint64_t getEndingNetId(DirectionHint hint) const;
	std::vector<Net*> getElementAdjacentNets(const Element* const element);

public:
	cv::Mat ciructImage() const;
	void detectElements(Yolo5* yolo);
	void detectNets(DirectionHint hint = C_DIRECTION_UNKOWN);
	std::string getString(DirectionHint hint = C_DIRECTION_UNKOWN);
	~Circut();
};
