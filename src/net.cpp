#include "net.h"

#include <cstdint>
#include <opencv2/imgproc.hpp>

#include "randomgen.h"
#include "utils.h"

Net::Net()
{
	id = rd::uid();
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
	cv::Rect paddedRect = padRect(element->getRect(), padding.first, padding.second, 5*tolleranceFactor);

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

uint64_t Net::getId() const
{
	return id;
}

bool Net::operator==(const Net& net) const
{
	return net.getId() == id;
}

size_t Net::elementCount() const
{
	return elements.size();
}
