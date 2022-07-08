#include "circut.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "log.h"
#include "randomgen.h"
#include "utils.h"

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

void Circut::getElements(Yolo5* yolo)
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
