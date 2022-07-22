#include "circut.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "log.h"
#include "randomgen.h"
#include "utils.h"
#include "linedetection.h"

cv::Mat Circut::ciructImage() const
{
	cv::Mat visulization;
	image.copyTo(visulization);
	for(size_t i = 0; i < elements.size(); ++i)
	{
		auto padding = getRectXYPaddingPercents(C_DIRECTION_UNKOWN, 1);
		cv::rectangle(visulization, padRect(elements[i].getRect(), padding.first, padding.second, 5), cv::Scalar(0,0,255), 2);
		cv::rectangle(visulization, elements[i].getRect(), cv::Scalar(0,255,255), 1);
		std::string labelStr = std::to_string(static_cast<int>(elements[i].getType())) +
			" P: " +  std::to_string(elements[i].getProb());
		cv::putText(visulization, labelStr,
			cv::Point(elements[i].getRect().x, elements[i].getRect().y-3),
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
			Element element(static_cast<ElementType>(detection.classId), detection.rect, detection.prob);
			element.image = image(detection.rect);
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
		clipLinesAgainstRect(lines, element.getRect());

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

Element* Circut::findUaccountedPathStartingElement(DirectionHint hint, size_t start, size_t stop, std::vector<const Element*>& handled)
{
	for(Element* element : nets[start].elements)
	{
		if(std::find(handled.begin(), handled.end(), element) != handled.end())
			continue;

		size_t opposingIndex = getOpositNetIndex(element, &nets[start]);
		if(opposingIndex == stop)
			return element;

		if(((hint == C_DIRECTION_HORIZ || hint == C_DIRECTION_UNKOWN) && (nets[start].center().x > nets[opposingIndex].center().x) ||
			(hint == C_DIRECTION_VERT && (nets[start].center().y > nets[opposingIndex].center().y))))
			continue;

		if(findUaccountedPathStartingElement(hint, opposingIndex, stop, handled) != nullptr)
			return element;
	}

	return nullptr;
}

size_t Circut::appendStringForSerisPath(std::string& str, const Element* element, std::vector<const Element*>& handled, size_t netIndex, size_t endNetIndex, size_t startNetIndex)
{
	int64_t opposing = getOpositNetIndex(element, &nets[netIndex]);

	if(element->getType() != E_TYPE_SOURCE && opposing >= 0)
	{
		std::string ch = element->getString();
		Log(Log::SUPERDEBUG)<<"adding "<<ch;
		str.append(ch);
		handled.push_back(element);
	}

	if(opposing < 0)
	{
		Log(Log::WARN)<<"Dangling element! return";
		return startNetIndex;
	}
	else if(nets[opposing].elements.size() > 2 || opposing == static_cast<int64_t>(endNetIndex) && nets[opposing].elements.size() == 2)
	{
		str.push_back(')');
		Log(Log::WARN)<<"Finished Series return";
		return opposing;
	}
	else if(opposing == static_cast<int64_t>(endNetIndex))
	{
		Log(Log::SUPERDEBUG)<<"path finished return";
		return endNetIndex;
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

		appendStringForSerisPath(str, elementL, handled, opposing, endNetIndex, startNetIndex);
	}
}

void Circut::appendStringForParalellPath(std::string& str, const Element* element, std::vector<const Element*>& handled, size_t netIndex, size_t endNetIndex, size_t startNetIndex)
{
	str.push_back('(');
	size_t seriesEndIndex = appendStringForSerisPath(str, element, handled, netIndex, endNetIndex, startNetIndex);

	if(seriesEndIndex != endNetIndex && seriesEndIndex != startNetIndex)
	{
		findUaccountedPathStartingElement(C_DIRECTION_UNKOWN, netIndex, seriesEndIndex, handled);
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

		Log(Log::DEBUG, false)<<"Connection count for "<<element.getString()<<" is "<<ajdacentNets.size();
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
		str.push_back('(');
		appendStringForSerisPath(str, element, handledElements, startingIndex, endIndex, startingIndex);
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
			cv::Point(element->getRect().x+offset, element->getRect().y+element->getRect().height+5),
			          cv::FONT_HERSHEY_PLAIN, 0.75, cv::Scalar(0,0,0), 1, cv::LINE_8, false);
			offset+=5;
		}
		cv::imshow("Viewer", visulization);
		cv::waitKey(0);
	}

	return model;
}
