#include "net.h"

#include <cstdint>
#include <opencv2/imgproc.hpp>

#include "randomgen.h"
#include "utils.h"
#include "log.h"

Net::Net()
{
	id = rd::uid();
}

Net::Net(cv::Point a, cv::Point b)
{
	Net();
	endpoints.push_back(a);
	endpoints.push_back(b);
	lines.push_back(cv::Vec4f(a.x, a.y, b.x, b.y));
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

size_t Net::indexOfEndpoint(const cv::Point2i& point)
{
	for(size_t i = 0; i < endpoints.size(); ++i)
	{
		if(endpoints[i] == point)
			return i;
	}
	assert(false);
	return 0;
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

double Net::closestEndpointDist(const cv::Point2i& point)
{
	assert(endpoints.size() > 0);
	std::vector<double> dists;
	dists.reserve(endpoints.size());
	for(const cv::Point2i& endpoint : endpoints)
		dists.push_back(pointDist(endpoint, point));
	return *std::min(dists.begin(), dists.end());
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

bool Net::removeElement(Element* element)
{
	auto iter = std::find(elements.begin(), elements.end(), element);

	if(iter != elements.end())
	{
		size_t index = iter-elements.begin();
		elements.erase(iter);
		connectedEndpointIndecies.erase(connectedEndpointIndecies.begin()+index);
		return true;
	}
	else
	{
		return false;
	}
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

std::vector<cv::Point2i> Net::unconnectedEndPoints() const
{
	std::vector<cv::Point2i> out;
	for(size_t i = 0; i < endpoints.size(); ++i)
	{
		if(std::find(connectedEndpointIndecies.begin(), connectedEndpointIndecies.end(), i) == connectedEndpointIndecies.end())
			out.push_back(endpoints[i]);
	}
	return out;
}

bool Net::mergeNet(Net& out, Net& net, double endpointTollerance,
				   DirectionHint hint, const cv::Point2i ignore) const
{
	out = *this;
	out.lines.insert(out.lines.end(), net.lines.begin(), net.lines.end());
	out.nodes.insert(out.nodes.end(), net.nodes.begin(), net.nodes.end());

	std::vector<Element*> joinedElements = out.elements;
	joinedElements.insert(joinedElements.end(), net.elements.begin(), net.elements.end());
	for(size_t i = 0; i < joinedElements.size(); ++i)
	{
		for(size_t j = 0; j < joinedElements.size(); ++j)
		{
			if(i == j)
				continue;
			if(joinedElements[i] == joinedElements[j])
			{
				joinedElements.erase(joinedElements.begin()+j);
				--i;
				break;
			}
		}
	}

	std::vector<cv::Point2i> unconnectedEpA = out.unconnectedEndPoints();
	std::vector<cv::Point2i> unconnectedEpB = net.unconnectedEndPoints();

	bool merged = false;
	for(size_t i = 0; i < unconnectedEpA.size(); ++i)
	{
		if(unconnectedEpA[i] == ignore)
			continue;
		for(size_t j = 0; j < unconnectedEpB.size(); ++j)
		{
			if(unconnectedEpB[i] == ignore)
				continue;
			if(pointDist(unconnectedEpA[i], unconnectedEpB[j]) < endpointTollerance)
			{
				out.nodes.push_back(unconnectedEpA[i]);
				size_t indexA = out.indexOfEndpoint(unconnectedEpA[i]);
				size_t indexB = net.indexOfEndpoint(unconnectedEpB[j]);
				out.endpoints.erase(out.endpoints.begin()+indexA);
				net.endpoints.erase(net.endpoints.begin()+indexB);
				unconnectedEpA.erase(unconnectedEpA.begin()+i);
				unconnectedEpB.erase(unconnectedEpB.begin()+j);
				--i;
				merged = true;
				break;
			}
		}
	}

	if(!merged)
	{
		Log(Log::WARN)<<"Unable to merge nets, no overlaping unconnected endpoints";
		return false;
	}

	out.endpoints.insert(out.endpoints.end(), net.endpoints.begin(), net.endpoints.end());
	out.elements.clear();
	for(Element* element : joinedElements)
	{
		bool ret = out.addElement(element, hint);
		if(!ret)
			Log(Log::WARN)<<"Unable to re-add element "<<element->getString()<<" at "
			<<element->getRect().x<<'x'<<element->getRect().y;
	}

	return true;
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
