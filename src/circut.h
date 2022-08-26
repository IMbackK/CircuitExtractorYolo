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
	cv::Rect rect;
	cv::Mat image;
	std::vector<Element*> elements;
	std::vector<Net> nets;
	DirectionHint dirHint = C_DIRECTION_UNKOWN;

private:
	static bool moveConnectedLinesIntoNet(Net& net, size_t index, std::vector<cv::Vec4f>& lines, double tollerance);
	static std::vector<Net> sortLinesIntoNets(std::vector<cv::Vec4f> lines, double tollerance);
	static Net* netFromId(std::vector<Net>& nets, uint64_t id);
	static bool colapseSerial(std::vector<Net>& nets, std::vector<Element*>& joinedElements, uint64_t startingId, uint64_t endingId);
	static bool colapseParallel(std::vector<Net>& nets, std::vector<Element*>& joinedElements, uint64_t startingId, uint64_t endingId);
	static uint64_t getOpositNetId(const Element* element, const Net& net, const std::vector<Net>& netsL);
	static void dropUnessecaryBrakets(std::string& str);

	void removeUnconnectedNets();
	void removeNestedElements();
	bool healDanglingElement(Element* element);
	uint64_t getStartingNetId(DirectionHint hint) const;
	uint64_t getEndingNetId(DirectionHint hint) const;
	std::vector<Net*> getElementAdjacentNets(const Element* const element);

public:
	Circut() = default;
	Circut(const Circut& in);
	Circut(cv::Mat imageI, float probI, cv::Rect rectI);
	Circut operator=(const Circut& in);
	cv::Mat ciructImage() const;
	void detectElements(Yolo5* yolo);
	void setDirectionHint(DirectionHint hint);
	void detectNets();
	bool parseCircut();
	std::string getString();
	std::string getSummary();
	~Circut();
	DirectionHint estimateDirection();
};
