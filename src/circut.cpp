#include "circut.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "log.h"
#include "randomgen.h"
#include "utils.h"
#include "linedetection.h"

static std::pair<double, double> getRectXYPaddingPercents(DirectionHint hint)
{
	double padX;
	double padY;
	switch(hint)
	{
		case C_DIRECTION_HORIZ:
			padX = 0.1;
			padY = 0;
			break;
		case C_DIRECTION_VERT:
			padX = 0;
			padY = 0.1;
			break;
		case C_DIRECTION_UNKOWN:
		default:
			padX = 0.1;
			padY = 0.05;
			break;
	}
	return std::pair<double, double>(padX, padY);
}

void Net::draw(cv::Mat& image) const
{
	cv::Scalar color(rd::rand(255), rd::rand(255), rd::rand(255));
	drawLineSegments(image, lines, color);

	for(const cv::Point2i& point : endpoints)
		cv::circle(image, point, 5, color, -1);

	for(const cv::Point2i& point : nodes)
		cv::circle(image, point, 5, color, 1);
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

bool Net::addElement(Element* element, DirectionHint hint)
{
	std::pair<double, double> padding = getRectXYPaddingPercents(hint);
	cv::Rect paddedRect = padRect(element->rect, padding.first, padding.second);

	for(const cv::Point2i& point : endpoints)
	{
		if(pointInRect(point, paddedRect))
		{
			elements.push_back(element);
			return true;
		}
	}
	return false;
}

cv::Mat Circut::ciructImage() const
{
	cv::Mat visulization;
	image.copyTo(visulization);
	for(size_t i = 0; i < elements.size(); ++i)
	{
		cv::rectangle(visulization, elements[i].rect, cv::Scalar(0,0,255), 2);
		std::string labelStr = std::to_string(static_cast<int>(elements[i].type)) +
			" P: " +  std::to_string(elements[i].prob);
		cv::putText(visulization, labelStr,
			cv::Point(elements[i].rect.x, elements[i].rect.y-3),
			cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255,0,0), 1, cv::LINE_8, false);
	}
	return visulization;
}

void Circut::detectElements(Yolo5* yolo)
{
	std::vector<Yolo5::DetectedClass> detections = yolo->detect(image);
	Log(Log::DEBUG)<<"Elements: "<<detections.size();

	if(detections.size() > 0 && Log::level == Log::SUPERDEBUG)
	{
		cv::Mat visulization;
		image.copyTo(visulization);
		for(const Yolo5::DetectedClass& detection : detections)
			Yolo5::drawDetection(visulization, detection);
		cv::imshow("Viewer", visulization);
		cv::waitKey(0);
	}

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

	std::pair<double, double> padding = getRectXYPaddingPercents(hint);

	for(const Element& element : elements)
		clipLinesAgainstRect(lines, padRect(element.rect, padding.first, padding.second));

	nets = sortLinesIntoNets(lines, std::max(image.rows/15.0, 5.0));

	for(Net& net : nets)
	{
		net.computePoints(std::max(image.rows/15.0, 5.0));
	}
}

void Circut::parseString(DirectionHint hint)
{
	for(Net& net : nets)
	{
		for(Element& element : elements)
			net.addElement(&element, hint);
	}


}
