#include "circut.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "log.h"

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
