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
	std::vector<Element*> elements;
	cv::Rect rect;
	std::vector<Net> nets;
	size_t pagenum;

public:

	float prob;
	cv::Mat image;
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
	bool healDanglingNet(Net& net);
	std::vector<Net*> healOverconnectedElement(Element* element, std::vector<Net*> ajdacentNets);
	uint64_t getStartingNetId(DirectionHint hint) const;
	uint64_t getEndingNetId(DirectionHint hint) const;
	std::vector<Net*> getElementAdjacentNets(const Element* const element);


public:
	Circut() = default;
	Circut(const Circut& in);
	Circut(cv::Mat image, float prob, cv::Rect rect, size_t pagenum = 0);
	Circut operator=(const Circut& in);
	~Circut();
	cv::Mat ciructImage() const;
	cv::Mat plainCircutImage() const;
	void detectElements(Yolo5* yolo);
	const std::vector<Element*>& getElements() const;
	void setDirectionHint(DirectionHint hint);
	void detectNets();
	bool parseCircut();
	void dropImage();
	cv::Rect getRect() const;
	std::string getString();
	std::string getSummary();
	std::string getYoloElementLabels() const;
	size_t getPagenum() const {return pagenum;}
	void setPagenum(size_t pagenumI) {pagenum = pagenumI;}
	DirectionHint estimateDirection();
};
