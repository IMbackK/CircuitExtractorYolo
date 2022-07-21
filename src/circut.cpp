#include "circut.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "log.h"
#include "randomgen.h"
#include "utils.h"
#include "linedetection.h"

char Element::getChar() const
{
	switch(type)
	{
		case E_TYPE_R:
			return 'r';
		case E_TYPE_C:
			return 'c';
		case E_TYPE_L:
			return 'l';
		case E_TYPE_P:
			return 'p';
		case E_TYPE_W:
			return 'w';
		case E_TYPE_SOURCE:
			return 's';
		case E_TYPE_UNKOWN:
		default:
			return 'x';
	}
}

static std::pair<double, double> getRectXYPaddingPercents(DirectionHint hint, double tolleranceFactor)
{
	double padX;
	double padY;
	switch(hint)
	{
		case C_DIRECTION_HORIZ:
			padX = 0.15*tolleranceFactor;
			padY = 0;
			break;
		case C_DIRECTION_VERT:
			padX = 0;
			padY = 0.15*tolleranceFactor;
			break;
		case C_DIRECTION_UNKOWN:
		default:
			padX = 0.15*tolleranceFactor;
			padY = 0.075*tolleranceFactor;
			break;
	}
	return std::pair<double, double>(padX, padY);
}

void Net::draw(cv::Mat& image, const cv::Scalar* color) const
{
	cv::Scalar colorFinal = color ? *color : cv::Scalar(rd::rand(64)+64, rd::rand(191)+64, rd::rand(64)+64);
	drawLineSegments(image, lines, colorFinal);

	for(size_t i = 0; i < endpoints.size(); ++i)
	{
		auto iter = std::find(connectedEndpointIndecies.begin(), connectedEndpointIndecies.end(), i);
		if(iter != connectedEndpointIndecies.end())
			cv::drawMarker(image, endpoints[i], colorFinal, cv::MARKER_SQUARE, 10, 1);
		else
			cv::circle(image, endpoints[i], 5, colorFinal, -1);
	}

	for(const cv::Point2i& point : nodes)
		cv::circle(image, point, 5, colorFinal, 1);
}

bool Net::pointIsFree(const cv::Point2i& point, const size_t ignore, double tollerance)
{
	bool freePoint = true;
	for(size_t j = 0; j < lines.size(); ++j)
	{
		if(ignore == j)
			continue;
		if(pointIsOnLine(point, lines[j], tollerance))
		{
			freePoint = false;
			break;
		}
	}
	return freePoint;
}

void Net::computePoints(double tollerance)
{
	for(size_t i = 0; i < lines.size(); ++i)
	{
		const cv::Point2i start(lines[i][0], lines[i][1]);
		const cv::Point2i end(lines[i][2], lines[i][3]);

		if(pointIsFree(start, i, tollerance))
			endpoints.push_back(start);
		else
			nodes.push_back(start);
		if(pointIsFree(end, i, tollerance))
			endpoints.push_back(end);
		else
			nodes.push_back(end);
	}

	deduplicatePoints(nodes, tollerance);
}

void Net::coordScale(double factor)
{
	for(cv::Point2i& point : endpoints)
	{
		point.x *= factor;
		point.y *= factor;
	}

	for(cv::Point2i& point : nodes)
	{
		point.x *= factor;
		point.y *= factor;
	}

	for(cv::Vec4f& vec : lines)
	{
		vec[0] *= factor;
		vec[1] *= factor;
		vec[2] *= factor;
		vec[3] *= factor;
	}
}

bool Net::addElement(Element* element, DirectionHint hint, double tolleranceFactor)
{
	std::pair<double, double> padding = getRectXYPaddingPercents(hint, tolleranceFactor);
	cv::Rect paddedRect = padRect(element->rect, padding.first, padding.second, 5*tolleranceFactor);

	for(size_t i = 0; i < endpoints.size(); ++i)
	{
		const cv::Point2i& point = endpoints[i];
		if(pointInRect(point, paddedRect))
		{
			elements.push_back(element);
			connectedEndpointIndecies.push_back(i);
			return true;
		}
	}
	return false;
}

cv::Rect Net::endpointRect() const
{
	int left = std::numeric_limits<int>::max();
	int right = 0;
	int top = std::numeric_limits<int>::max();
	int bottom = 0;

	for(size_t i = 0; i < endpoints.size(); ++i)
	{
		left = endpoints[i].x < left ? endpoints[i].x : left;
		right = endpoints[i].x > right ? endpoints[i].x : right;
		top = endpoints[i].y < top ? endpoints[i].y : top;
		bottom = endpoints[i].y > bottom ? endpoints[i].y : bottom;
	}
	return cv::Rect(left, top, right-left, bottom-top);
}

cv::Point Net::center() const
{
	cv::Rect rect = endpointRect();
	return cv::Point(rect.x+rect.width/2, rect.y+rect.height/2);
}

cv::Mat Circut::ciructImage() const
{
	cv::Mat visulization;
	image.copyTo(visulization);
	for(size_t i = 0; i < elements.size(); ++i)
	{
		auto padding = getRectXYPaddingPercents(C_DIRECTION_UNKOWN, 1);
		cv::rectangle(visulization, padRect(elements[i].rect, padding.first, padding.second, 5), cv::Scalar(0,0,255), 2);
		cv::rectangle(visulization, elements[i].rect, cv::Scalar(0,255,255), 1);
		std::string labelStr = std::to_string(static_cast<int>(elements[i].type)) +
			" P: " +  std::to_string(elements[i].prob);
		cv::putText(visulization, labelStr,
			cv::Point(elements[i].rect.x, elements[i].rect.y-3),
			cv::FONT_HERSHEY_PLAIN, 0.75, cv::Scalar(255,0,0), 1, cv::LINE_8, false);
	}

	size_t firstNetIndex = getStartingIndex(C_DIRECTION_UNKOWN);
	size_t lastNetIndex = getEndingIndex(C_DIRECTION_UNKOWN);

	for(size_t i = 0; i < nets.size(); ++i)
	{
		if(i == firstNetIndex)
		{
			cv::Scalar color(255,0,0);
			nets[i].draw(visulization, &color);
		}
		else if(i == lastNetIndex)
		{
			cv::Scalar color(0,0,255);
			nets[i].draw(visulization, &color);
		}
		else
		{
			nets[i].draw(visulization);
		}
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
			Element element;
			element.image = image(detection.rect);
			element.type = static_cast<ElementType>(detection.classId);
			element.rect = detection.rect;
			element.prob = detection.prob;
			elements.push_back(element);
		}
		catch(const cv::Exception& ex)
		{
			Log(Log::WARN)<<detection.rect<<" out of bounds";
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
		Log(Log::SUPERDEBUG)<<"---NEW NET---";
		net.lines.push_back(*lines.begin());
		lines.erase(lines.begin());
		while(moveConnectedLinesIntoNet(net, net.lines.size()-1, lines, tollerance));
		nets.push_back(net);
	}
	return nets;
}

void Circut::detectNets(DirectionHint hint)
{
	assert(image.data);

	std::vector<cv::Vec4f> lines = lineDetect(image);

	for(const Element& element : elements)
		clipLinesAgainstRect(lines, element.rect);

	nets = sortLinesIntoNets(lines, std::max(image.rows/15.0, 5.0));

	for(Net& net : nets)
	{
		net.computePoints(std::max(image.rows/15.0, 5.0));
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

size_t Circut::getStartingIndex(DirectionHint hint) const
{
	size_t leftMostIndex = 0;
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
	return leftMostIndex;
}

size_t Circut::getEndingIndex(DirectionHint hint) const
{
	size_t rightMostIndex = 0;
	int rightMostPoint = 0;
	for(size_t i = 0; i < nets.size(); ++i)
	{
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
	return rightMostIndex;
}

int64_t Circut::getOpositNetIndex(const Element* element, Net* net) const
{
	for(size_t i = 0; i < nets.size(); ++i)
	{
		if(&nets[i] == net)
			continue;

		for(size_t j = 0; j < nets[i].elements.size(); ++j)
		{
			if(element == nets[i].elements[j])
				return static_cast<int64_t>(i);
		}
	}
	return -1;
}
handled
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

void Circut::getStringForPath(std::string& str, const Element* element, std::vector<const Element*>& handled, size_t netIndex, size_t endNetIndex, size_t startNetIndex)
{
	int64_t opposing = getOpositNetIndex(element, &nets[netIndex]);

	if(element->type != E_TYPE_SOURCE && opposing >= 0)
	{
		const char ch = element->getChar();
		Log(Log::SUPERDEBUG)<<"adding "<<ch;
		str.push_back(ch);
		handled.push_back(element);
	}

	if(opposing == static_cast<int64_t>(endNetIndex))
	{
		if(nets[netIndex].elements.size() > 2 || (startNetIndex == netIndex && nets[netIndex].elements.size() > 1))
		{
			Log(Log::SUPERDEBUG)<<"adding (";
			str.push_back('(');
		}
		Log(Log::SUPERDEBUG)<<"path finished return";
		return;
	}
	else if(opposing < 0)
	{
		Log(Log::WARN)<<"Dangling element! return";
		return;
	}

	str.push_back('-');

	for(const Element* elementL : nets[opposing].elements)
	{
		if(elementL == element)
			continue;

		if(std::find(handled.begin(), handled.end(), elementL) != handled.end())
		{
			str.push_back(')');
			continue;
		}

		getStringForPath(str, elementL, handled, opposing, endNetIndex, startNetIndex);
	}
}

void Circut::balanceBrackets(std::string& str)
{
	int bracketCount = 0;

	for(const char ch : str)
	{
		if(ch == '(')
			++bracketCount;
		else if(ch == ')')
			--bracketCount;
	}

	if(bracketCount < 0)
	{
		Log(Log::ERROR)<<"Invalid model string parsed for circut";
		assert(bracketCount >= 0);
	}

	for(int i = 0; i < bracketCount; ++i)
	{
		str.push_back(')');
	}
}

std::string Circut::getString(DirectionHint hint)
{
	if(!model.empty())
		return model;

	size_t startingIndex = getStartingIndex(hint);
	size_t endIndex = getEndingIndex(hint);

	for(Net& net : nets)
	{
		for(Element& element : elements)
			net.addElement(&element, hint);
	}

	for(Element& element : elements)
	{
		if(getElementAdjacentNets(&element).size() < 2)
		{
			for(Net& net : nets)
				net.addElement(&element, hint, 3);
		}
	}

	bool dangling = false;
	for(Element& element : elements)
	{
		std::vector<Net*> ajdacentNets = getElementAdjacentNets(&element);

		Log(Log::DEBUG, false)<<"Connection count for "<<element.getChar()<<" is "<<ajdacentNets.size();
		if(std::find(ajdacentNets.begin(), ajdacentNets.end(), &nets[startingIndex]) != ajdacentNets.end())
			Log(Log::DEBUG, false)<<" includes first net";
		if(std::find(ajdacentNets.begin(), ajdacentNets.end(), &nets[endIndex]) != ajdacentNets.end())
			Log(Log::DEBUG, false)<<" includes last net";
		Log(Log::DEBUG, false)<<'\n';

		if(getElementAdjacentNets(&element).size() < 2)
			dangling = true;
	}

	if(dangling)
		Log(Log::WARN)<<"A dangling element is present";
	else
		Log(Log::DEBUG)<<"All elements fully connected";

	removeUnconnectedNets();

	std::string str;
	std::vector<const Element*> handledElements;
	for(const Element* element : nets[startingIndex].elements)
	{
		if(std::find(handledElements.begin(), handledElements.end(), element) != handledElements.end())
			continue;
		getStringForPath(str, element, handledElements, startingIndex, endIndex, startingIndex);
	}
	balanceBrackets(str);
	model = str;

	if(Log::level == Log::SUPERDEBUG)
	{
		cv::Mat visulization = ciructImage();
		int offset = 0;
		for(size_t i = 0; i < handledElements.size(); ++i)
		{
			const Element* const element = handledElements[i];
			cv::putText(visulization, std::to_string(i),
			cv::Point(element->rect.x+offset, element->rect.y+element->rect.height+5),
			          cv::FONT_HERSHEY_PLAIN, 0.75, cv::Scalar(0,0,0), 1, cv::LINE_8, false);
			offset+=5;
		}
		cv::imshow("Viewer", visulization);
		cv::waitKey(0);
	}

	return model;
}
