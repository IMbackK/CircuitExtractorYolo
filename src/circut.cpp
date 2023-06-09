#include "circut.h"

#include <cstdint>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <sstream>
#include <thread>

#include "element.h"
#include "log.h"
#include "randomgen.h"
#include "utils.h"
#include "linedetection.h"

static bool pairCompare(const std::pair<double, size_t> a, const std::pair<double, size_t> b)
{
	return a.first < b.first;
}

Circut::Circut(const Circut& in)
{
	model = in.model;
	rect = in.rect;
	image = in.image;
	nets = in.nets;
	dirHint = in.dirHint;
	pagenum = in.pagenum;

	elements.resize(in.elements.size(), nullptr);
	for(size_t i = 0; i < elements.size(); ++i)
		elements[i] = new Element(*in.elements[i]);
}

Circut Circut::operator=(const Circut& in)
{
	return Circut(in);
}

Circut::Circut(cv::Mat imageI, float probI, cv::Rect rectI, size_t pagenumI):
image(imageI), prob(probI), rect(rectI), pagenum(pagenumI)
{
}

cv::Mat Circut::plainCircutImage() const
{
	return image;
}

cv::Rect Circut::getRect() const
{
	return rect;
}

cv::Mat Circut::ciructImage() const
{
	cv::Mat visulization;
	image.copyTo(visulization);
	for(size_t i = 0; i < elements.size(); ++i)
	{
		auto padding = getRectXYPaddingPercents(C_DIRECTION_UNKOWN, 1);
		cv::rectangle(visulization, padRect(elements[i]->getRect(), padding.first, padding.second, 5), cv::Scalar(0,0,255), 2);
		cv::rectangle(visulization, elements[i]->getRect(), cv::Scalar(0,255,255), 1);
		std::string labelStr = elements[i]->getString();
		cv::putText(visulization, labelStr,
			cv::Point(elements[i]->getRect().x, elements[i]->getRect().y-3),
			cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255,0,0), 1, cv::LINE_8, false);
	}

	uint64_t firstNetId = getStartingNetId(C_DIRECTION_UNKOWN);
	uint64_t lastNetId = getEndingNetId(C_DIRECTION_UNKOWN, firstNetId);

	for(size_t i = 0; i < nets.size(); ++i)
	{
		if(nets[i].getId() == firstNetId)
		{
			cv::Scalar color(255,0,0);
			nets[i].draw(visulization, &color);
		}
		else if(nets[i].getId() == lastNetId)
		{
			cv::Scalar color(0,0,255);
			nets[i].draw(visulization, &color);
		}
		else
		{
			nets[i].draw(visulization);
		}
	}

	cv::putText(visulization, std::string("Dir: ")+getDirectionString(dirHint), cv::Point(5, visulization.rows-5),
			cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(100,0,200), 1, cv::LINE_8, false);

	if(!model.empty())
	{
		cv::putText(visulization, std::string("Model: ")+model, cv::Point(5, visulization.rows-18),
			cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(100,0,200), 1, cv::LINE_8, false);
	}

	return visulization;
}

void Circut::detectElements(Yolo5* yolo)
{
	std::vector<Yolo5::DetectedClass> detections = yolo->detect(image);
	Log(Log::DEBUG)<<"Elements: "<<detections.size();

	for(const Yolo5::DetectedClass& detection : detections)
	{
		try
		{
			Element* element = new Element(static_cast<ElementType>(detection.classId), image(detection.rect), detection.rect, detection.prob);
			Log(Log::DEBUG)<<"Got element "<<element->getString();
			if(element->getType() == E_TYPE_SOURCE || element->getType() == E_TYPE_UNKOWN || element->getType() == E_TYPE_NODE)
			{
				Log(Log::DEBUG)<<"Ignoreing";
				continue;
			}
			elements.push_back(element);
		}
		catch(const cv::Exception& ex)
		{
			Log(Log::WARN)<<detection.rect<<" out of bounds";
		}
	}

	for(size_t i = 0; i < elements.size(); ++i)
	{
		for(size_t j = 0; j < elements.size(); ++j)
		{
			if(i == j)
				continue;
			if(rectsFullyOverlap(elements[i]->getRect(), elements[j]->getRect()))
			{
				if(elements[i]->getRect().width*elements[i]->getRect().height > elements[i]->getRect().width*elements[i]->getRect().height)
					elements.erase(elements.begin()+j);
				else
					elements.erase(elements.begin()+i);
				--i;
				break;
			}
		}
	}
}

bool Circut::moveConnectedLinesIntoNet(Net& net, size_t index, std::vector<cv::Vec4f>& lines, double tollerance)
{
	bool ret = false;
	for(size_t j = 0; j < lines.size();)
	{
		if(lineCrossesOrtho(net.lines[index], lines[j], tollerance))
		{
			Log(Log::SUPERDEBUG)<<"Checking for "<<index<<": "<<net.lines[index]<<"\t"<<lines[j]<<" matches";
			net.lines.push_back(lines[j]);
			lines.erase(lines.begin()+j);
			moveConnectedLinesIntoNet(net, net.lines.size()-1, lines, tollerance);
			j = 0;
			ret = true;
		}
		else
		{
			Log(Log::SUPERDEBUG)<<"Checking for "<<index<<": "<<net.lines[index]<<"  "<<lines[j]<<" dose not match";
			++j;
		}
	}
	Log(Log::SUPERDEBUG)<<"return "<<index;
	return ret;
}

std::vector<Net> Circut::sortLinesIntoNets(std::vector<cv::Vec4f> lines, double tollerance)
{
	std::vector<Net> nets;

	Log(Log::SUPERDEBUG)<<"Lines:";
	for(const cv::Vec4f& line : lines )
		Log(Log::SUPERDEBUG)<<line;

	while(!lines.empty())
	{
		Net net;
		Log(Log::SUPERDEBUG)<<"---NEW NET "<<net.getId()<<" ---";
		net.lines.push_back(*lines.begin());
		lines.erase(lines.begin());
		while(moveConnectedLinesIntoNet(net, net.lines.size()-1, lines, tollerance));
		nets.push_back(net);
	}
	return nets;
}

void Circut::detectNets()
{
	assert(image.data);

	std::vector<cv::Vec4f> lines = lineDetect(image);

	for(const Element* element : elements)
		clipLinesAgainstRect(lines, element->getRect());

	double tollerance = std::max(image.rows/15.0, 5.0);
	nets = sortLinesIntoNets(lines, tollerance);
	Log(Log::DEBUG)<<__func__<<"using tollerance: "<<tollerance;
	for(Net& net : nets)
	{
		net.computePoints(tollerance);
	}
}

void Circut::removeUnconnectedNets()
{
	for(size_t i = 0; i < nets.size(); ++i)
	{
		if(nets[i].elements.empty())
		{
			nets.erase(nets.begin()+i);
			--i;
		}
	}
}

uint64_t Circut::getStartingNetId(DirectionHint hint) const
{
	if(nets.empty())
		return 0;

	int leftMostIndex = 0;
	int leftMostPoint = std::numeric_limits<int>::max();
	for(size_t i = 0; i < nets.size(); ++i)
	{
		cv::Rect rect = nets[i].endpointRect();
		if((hint == C_DIRECTION_HORIZ || hint == C_DIRECTION_UNKOWN) && rect.x < leftMostPoint)
		{
			leftMostPoint = rect.x;
			leftMostIndex = i;
		}
		else if(hint == C_DIRECTION_VERT && rect.y < leftMostPoint)
		{
			leftMostPoint = rect.y;
			leftMostIndex = i;
		}
	}
	return nets[leftMostIndex].getId();

}

uint64_t Circut::getEndingNetId(DirectionHint hint, uint64_t ignoreNet) const
{
	if(nets.empty())
		return 0;

	int rightMostIndex = 0;
	int rightMostPoint = 0;
	for(size_t i = 0; i < nets.size(); ++i)
	{
		if(nets[i].getId() == ignoreNet)
			continue;
		cv::Rect rect = nets[i].endpointRect();
		if((hint == C_DIRECTION_HORIZ || hint == C_DIRECTION_UNKOWN) && rect.x+rect.width > rightMostPoint)
		{
			rightMostPoint = rect.x+rect.width;
			rightMostIndex = i;
		}
		else if(hint == C_DIRECTION_VERT && rect.y+rect.height > rightMostPoint)
		{
			rightMostPoint = rect.y+rect.height;
			rightMostIndex = i;
		}
	}
	return nets[rightMostIndex].getId();
}

uint64_t Circut::getOpositNetId(const Element* element, const Net& net, const std::vector<Net>& netsL)
{
	for(size_t i = 0; i < netsL.size(); ++i)
	{
		if(netsL[i] == net)
			continue;

		for(size_t j = 0; j < netsL[i].elements.size(); ++j)
		{
			if(element == netsL[i].elements[j])
				return netsL[i].getId();
		}
	}
	return 0;
}

std::vector<Net*> Circut::getElementAdjacentNets(const Element* const element)
{
	std::vector<Net*> out;
	for(Net& net : nets)
	{
		for(const Element* netElement: net.elements)
		{
			if(element == netElement)
				out.push_back(&net);
		}
	}
	return out;
}

Net* Circut::netFromId(std::vector<Net>& netsL, uint64_t id)
{
	for(size_t i = 0; i < netsL.size(); ++i)
	{
		if(netsL[i].getId() == id)
			return &netsL[i];
	}
	return nullptr;
}

bool Circut::colapseSerial(std::vector<Net>& netsL, std::vector<Element*>& joinedElements, uint64_t startingId, uint64_t endingId)
{
	for(size_t i = 0; i < netsL.size(); ++i)
	{
		if(netsL[i].getId() == startingId || netsL[i].getId() == endingId)
			continue;
		if(netsL[i].elementCount() == 2)
		{
			Element* join = new Element(*netsL[i].elements[0], *netsL[i].elements[1], true);
			joinedElements.push_back(join);

			uint64_t oppositId = getOpositNetId(netsL[i].elements[0], netsL[i], netsL);
			Net* left = netFromId(netsL, oppositId);
			if(left)
			{
				std::erase(left->elements, netsL[i].elements[0]);
				std::erase(left->elements, netsL[i].elements[1]);
				left->elements.push_back(join);
			}
			else
			{
				Log(Log::ERROR)<<"Invalid net for id: "<<oppositId;
			}

			oppositId = getOpositNetId(netsL[i].elements[1], netsL[i], netsL);
			Net* right = netFromId(netsL, oppositId);
			if(right)
			{
				std::erase(right->elements, netsL[i].elements[0]);
				std::erase(right->elements, netsL[i].elements[1]);
				right->elements.push_back(join);
			}
			else
			{
				Log(Log::ERROR)<<"Invalid net for id: "<<oppositId;
			}

			netsL.erase(netsL.begin() + i);
			return true;
		}
	}
	return false;
}

bool Circut::healAdjacentDanglingElement(Element* element, bool multiWay)
{
	auto padding = getRectXYPaddingPercents(dirHint, 1);
	cv::Rect rect = padRect(element->getRect(), padding.first, padding.second, 2);
	std::vector<Net*> ajdacentNets = getElementAdjacentNets(element);
	bool horiz = (dirHint == C_DIRECTION_HORIZ || dirHint == C_DIRECTION_UNKOWN);

	for(Element* elementTest : elements)
	{
		if(element == elementTest)
			continue;

		if(!multiWay && getElementAdjacentNets(elementTest).size() > 1)
			continue;

		bool found = false;
		for(const Element* elementOnAjdecentNet : ajdacentNets[0]->elements)
		{
			if(elementTest == elementOnAjdecentNet)
			{
				found = true;
				break;
			}
		}
		if(found)
			continue;

		cv::Rect rectTest = padRect(elementTest->getRect(), padding.first, padding.second, 2);

		if(rectsIntersect(rect, rectTest))
		{
			if((horiz && ((rect.x >= rectTest.x && rect.x <= rectTest.x+rect.width) ||
				(rect.x+rect.width >= rectTest.x && rect.x+rect.width <= rectTest.x+rect.width))) ||
				(!horiz && ((rect.y >= rectTest.y && rect.y <= rectTest.y+rect.height) ||
				(rect.y+rect.height >= rectTest.y && rect.y+rect.height <= rectTest.y+rect.height))))
			{
				Log(Log::INFO)<<"joining adjecent dangling elements with new net";
				nets.push_back(Net(element->center(), elementTest->center()));
				nets.back().addElement(element);
				nets.back().addElement(elementTest);
				return true;
			}
		}
	}

	return false;
}

bool Circut::healDanglingElement(Element* element)
{
	Log(Log::WARN)<<"Element "<<element->getString()
	<<" at "<<rect.x<<'x'<<rect.y<<" is dangling, trying to heal";
	Net* startingNet = netFromId(nets, getStartingNetId(dirHint));
	Net* endingNet = netFromId(nets, getEndingNetId(dirHint, startingNet->getId()));
	std::vector<Net*> ajdacentNets = getElementAdjacentNets(element);

	bool horiz = (dirHint == C_DIRECTION_HORIZ || dirHint == C_DIRECTION_UNKOWN);
	cv::Rect elRect = element->getRect();
	cv::Point elPoint = element->center();

	if(*startingNet == *ajdacentNets[0])
	{
		bool isEndingElement = (horiz && elPoint.x <= startingNet->center().x) ||
			(!horiz && elPoint.y <= startingNet->center().y);
		if(isEndingElement)
		{
			Log(Log::INFO)<<"Healing circut that starts with element";

			cv::Point a;
			cv::Point b;
			if(horiz)
			{
				a = cv::Point(elRect.x-elRect.width, elPoint.y);
				b = cv::Point(elRect.x, elPoint.y);
			}
			else
			{
				a = cv::Point(elPoint.x, elRect.y-elRect.height);
				b = cv::Point(elPoint.x, elRect.y);
			}

			nets.push_back(Net(a, b));

			nets.back().addElement(element);
			return true;
		}
	}
	else if(*endingNet == *ajdacentNets[0])
	{
		bool isEndingElement = (horiz && elPoint.x >= endingNet->center().x) ||
			(!horiz && elPoint.y >= endingNet->center().y);
		if(isEndingElement)
		{
			Log(Log::INFO)<<"Healing circut that ends with element";

			cv::Point a;
			cv::Point b;
			if(horiz)
			{
				a = cv::Point(elRect.x+elRect.width*2, elPoint.y);
				b = cv::Point(elRect.x+elRect.width, elPoint.y);
			}
			else
			{
				a = cv::Point(elPoint.x, elRect.y+elRect.height*2);
				b = cv::Point(elPoint.x, elRect.y+elRect.width);
			}

			nets.push_back(Net(a, b));

			nets.back().addElement(element);
			return true;
		}
	}
	else
	{
		bool ret = healAdjacentDanglingElement(element, false);
		if(!ret)
			ret = healAdjacentDanglingElement(element, true);
		return ret;
	}

	return false;
}

bool Circut::colapseParallel(std::vector<Net>& netsL, std::vector<Element*>& joinedElements, uint64_t startingId, uint64_t endingId)
{
	for(size_t i = 0; i < netsL.size(); ++i)
	{
		if(netsL[i].elementCount() > 2 ||
			(netsL[i].getId() == startingId && netsL[i].elementCount() == 2) ||
			(netsL[i].getId() == endingId && netsL[i].elementCount() == 2))
		{
			for(Element* elementA : netsL[i].elements)
			{
				for(Element* elementB : netsL[i].elements)
				{
					if(elementA == elementB)
						continue;

					uint64_t idElA = getOpositNetId(elementA, netsL[i], netsL);
					Net* opposingNet = netFromId(netsL, idElA);
					if(opposingNet && idElA == getOpositNetId(elementB, netsL[i], netsL))
					{
						Element* join = new Element(*elementA, *elementB, false);
						joinedElements.push_back(join);
						std::erase(netsL[i].elements, elementA);
						std::erase(netsL[i].elements, elementB);
						std::erase(opposingNet->elements, elementA);
						std::erase(opposingNet->elements, elementB);

						netsL[i].elements.push_back(join);
						opposingNet->elements.push_back(join);
						return true;
					}
				}
			}
		}
	}
	return false;
}

void Circut::dropUnessecaryBrakets(std::string& str)
{
	if(str.size() < 3)
		return;
	for(size_t i = 0; i < str.size()-2; ++i)
	{
		if(str[i] == '(' && str[i+2] == ')')
		{
			str.erase(str.begin()+i+2);
			str.erase(str.begin()+i);
			--i;
		}
	}
}

bool Circut::parseCircut()
{
	if(nets.empty())
	{
		Log(Log::WARN)<<"Can't parse circut without nets";
		return false;
	}

	if(elements.empty())
	{
		Log(Log::WARN)<<"Can't parse circut without elements";
		return false;
	}

	for(Net& net : nets)
	{
		for(Element* element : elements)
			net.addElement(element, dirHint);
	}

	for(Element* element : elements)
	{
		if(getElementAdjacentNets(element).size() < 2)
		{
			for(Net& net : nets)
				net.addElement(element, dirHint, 3);
		}
	}

	removeUnconnectedNets();

	uint64_t startingNetId = getStartingNetId(dirHint);
	uint64_t endingNetId = getEndingNetId(dirHint, startingNetId);

	Log(Log::DEBUG)<<"starting net: "<<startingNetId<<" ending net: "<<endingNetId;

	for(size_t i = 0; i < nets.size(); ++i)
	{
		std::vector<cv::Point2i> unconnected = nets[i].unconnectedEndPoints();
		if(unconnected.size() > 0 && nets[i].getId() != startingNetId && nets[i].getId() != endingNetId)
		{
			Log(Log::WARN)<<"Net "<<i<<" id:"<<nets[i].getId()<<
				" has too many unconnected endpoints, trying to heal";
			Log(Log::DEBUG)<<"Unconnected endpoints:";
			for(const cv::Point2i& point : unconnected)
				Log(Log::DEBUG)<<point;

			if(healDanglingNet(nets[i]))
				--i;
		}
	}

	bool dangling = false;
	for(size_t i = 0; i < elements.size(); ++i)
	{
		std::vector<Net*> ajdacentNets = getElementAdjacentNets(elements[i]);

		Log(Log::DEBUG, false)<<"Connection count for "<<elements[i]->getString()<<" is "<<ajdacentNets.size();
		if(std::find(ajdacentNets.begin(), ajdacentNets.end(), netFromId(nets, startingNetId)) != ajdacentNets.end())
			Log(Log::DEBUG, false)<<" includes first net";
		if(std::find(ajdacentNets.begin(), ajdacentNets.end(), netFromId(nets, endingNetId)) != ajdacentNets.end())
			Log(Log::DEBUG, false)<<" includes last net";
		Log(Log::DEBUG, false)<<'\n';

		for(Net* net : ajdacentNets)
		{
			Log(Log::DEBUG)<<net->getId();
		}

		if(ajdacentNets.size() == 0)
		{
			Log(Log::WARN)<<"Deleteing wholy unconnected element "
				<<elements[i]->getString()<<" at "<<elements[i]->getRect().x<<'x'<<elements[i]->getRect().y;
			delete elements[i];
			elements.erase(elements.begin()+i);
			--i;
		}
		else if(ajdacentNets.size() > 2)
		{
			ajdacentNets = healOverconnectedElement(elements[i], ajdacentNets);
		}
	}

	for(size_t i = 0; i < elements.size(); ++i)
	{
		std::vector<Net*> ajdacentNets = getElementAdjacentNets(elements[i]);
		if(ajdacentNets.size() == 1)
		{
			dangling |= !healDanglingElement(elements[i]);
		}
	}

	removeUnconnectedNets();

	if(elements.empty())
	{
		Log(Log::WARN)<<"All elements are dangling";
		return false;
	}

	if(dangling)
	{
		Log(Log::WARN)<<"A dangling element is present";
		return false;
	}
	else
	{
		Log(Log::DEBUG)<<"All elements fully connected";
	}

	return true;
}

bool Circut::healDanglingNet(Net& net)
{
	const double tollerance = std::max(image.rows/10.0, 10.0);
	std::vector<cv::Point2i> unconnectedEndPoints = net.unconnectedEndPoints();

	double minDist = std::numeric_limits<double>::max();
	ssize_t index = -1;
	for(size_t i = 0; i < nets.size(); ++i)
	{
		if(nets[i] == net)
			continue;
		for(const cv::Point2i& endpoint : unconnectedEndPoints)
		{
			double distance = nets[i].closestEndpointDist(endpoint);
			Log(Log::WARN)<<__func__<<endpoint<<" d:"<<distance;
			if(minDist > distance)
			{
				index = i;
				minDist = distance;
			}
		}
	}

	if(index < 0 || tollerance < minDist)
		return false;

	Net joinedNet;
	bool ret = net.mergeNet(joinedNet, nets[index], tollerance, dirHint);
	if(ret)
	{
		net = joinedNet;
		nets.erase(nets.begin()+index);
	}

	return ret;
}

std::vector<Net*> Circut::healOverconnectedElement(Element* element, std::vector<Net*> ajdacentNets)
{
	assert(ajdacentNets.size() > 2);

	cv::Rect rect = element->getRect();

	Log(Log::WARN)<<"Element "<<element->getString()
	<<" at "<<rect.x<<'x'<<rect.y<<" has too manny connected nets, trying to heal";

	std::vector<std::pair<double, size_t>> aDists;
	std::vector<std::pair<double, size_t>> bDists;
	cv::Point2i pa;
	cv::Point2i pb;

	aDists.reserve(ajdacentNets.size());
	bDists.reserve(ajdacentNets.size());

	if(dirHint == C_DIRECTION_UNKOWN || dirHint == C_DIRECTION_HORIZ)
	{
		pa = cv::Point2i(rect.x, rect.y+rect.height/2);
		pb = cv::Point2i(rect.x+rect.width, rect.y+rect.height/2);
	}
	else
	{
		pa = cv::Point2i(rect.x+rect.width/2, rect.y);
		pb = cv::Point2i(rect.x+rect.width/2, rect.y+rect.height);
	}

	Log(Log::SUPERDEBUG)<<"Using endpoints "<<pa<<' '<<pb;

	for(size_t i = 0; i < ajdacentNets.size(); ++i)
	{
		for(const cv::Point2i& point : ajdacentNets[i]->endpoints)
		{
			Log(Log::SUPERDEBUG)<<point<<" : "<<i;
		}
		aDists.push_back({ajdacentNets[i]->closestEndpointDist(pa), i});
		bDists.push_back({ajdacentNets[i]->closestEndpointDist(pb), i});
	}

	std::sort(aDists.begin(), aDists.end(), pairCompare);
	std::sort(bDists.begin(), bDists.end(), pairCompare);

	Log(Log::SUPERDEBUG)<<"aDists";
	for(const std::pair<double, size_t>& dist : aDists)
	{
		Log(Log::SUPERDEBUG)<<dist.first<<" : "<<dist.second;
	}

	Log(Log::SUPERDEBUG)<<"bDists";
	for(const std::pair<double, size_t>& dist : bDists)
	{
		Log(Log::SUPERDEBUG)<<dist.first<<" : "<<dist.second;
	}

	size_t aIndex = aDists[0].second;
	size_t bIndex = bDists[0].second == aDists[0].second ? bDists[1].second : bDists[0].second;

	for(size_t j = 0; j < ajdacentNets.size(); ++j)
	{
		if(j != aIndex && j != bIndex)
		{
			bool ret = ajdacentNets[j]->removeElement(element);
			assert(ret);
		}
	}

	return getElementAdjacentNets(element);
}

std::string Circut::getString()
{
	if(nets.size() < 2)
	{
		Log(Log::WARN)<<"Can't generate string for circut without at least two nets";
		return "";
	}

	if(elements.empty())
	{
		Log(Log::WARN)<<"Can't generate string for circut without elements";
		return "";
	}

	if(!model.empty())
		return model;

	uint64_t startingNetId = getStartingNetId(dirHint);
	uint64_t endingNetId = getEndingNetId(dirHint, startingNetId);

	std::string str;
	std::vector<Net> netsTmp = nets;
	std::vector<Element*> joinedElements;

	while(true)
	{
		bool progress = false;
		while(true)
		{
			bool ret = colapseSerial(netsTmp, joinedElements, startingNetId, endingNetId);
			if(ret)
				progress = true;
			if(!ret)
				break;
		}

		while(true)
		{
			bool ret = colapseParallel(netsTmp, joinedElements, startingNetId, endingNetId);
			if(ret)
				progress = true;
			if(!ret)
				break;
		}
		if(!progress)
			break;
	}

	if(!netsTmp.empty() && !netsTmp[0].elements.empty())
	{
		model = netsTmp[0].elements[0]->getString();
	}
	else
	{
		Log(Log::WARN)<<"Could not parse circut string";
		model = "";
	}

	for(Element* element : joinedElements)
		delete element;

	dropUnessecaryBrakets(model);

	return model;
}

std::string Circut::getSummary()
{
	std::stringstream ss;
	ss<<"Rect = "<<rect<<'\n';
	ss<<"Model = "<<getString()<<'\n';
	ss<<"Direction = "<<getDirectionString(dirHint);
	return ss.str();
}

std::string Circut::getYoloElementLabels() const
{
	std::stringstream ss;
	for(const Element* element : elements)
		ss<<yoloLabelsFromRect(element->getRect(), image, static_cast<int>(element->getType()));
	return ss.str();
}

const std::vector<Element*>& Circut::getElements() const
{
	return elements;
}

DirectionHint Circut::estimateDirection()
{
	if(nets.size() < 2)
	{
		Log(Log::WARN)<<"Can't estmate direction of circut with less than two nets";
		return C_DIRECTION_UNKOWN;
	}

	std::vector<cv::Point> points;
	points.reserve(nets.size());
	for(const Net& net : nets)
		points.push_back(net.center());
	cv::Rect rect = rectFromPoints(points);

	if(rect.width >= rect.height)
		return C_DIRECTION_HORIZ;
	else
		return C_DIRECTION_VERT;
}

void Circut::setDirectionHint(DirectionHint hint)
{
	dirHint = hint;
}

void Circut::dropImage()
{
	image.release();
}

Circut::~Circut()
{
	for(Element* element : elements)
		delete element;
}

