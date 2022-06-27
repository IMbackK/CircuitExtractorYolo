#include "yolo.h"
#include <vector>
#include <string>
#include <opencv2/imgproc.hpp>

#include "log.h"

static void printMatInfo(const cv::Mat& mat, const std::string& prefix = "")
{
	Log(Log::INFO)<<prefix;
	Log(Log::INFO)<<mat.rows<<'x'<<mat.cols<<" Channels: "<<mat.channels();
}

Yolo5::Yolo5(const cv::dnn::Net &netI, size_t numCassesI):
numClasses(numCassesI), net(netI)
{
	dimensions = 5+numClasses;
	net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
	net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

Yolo5::Yolo5(const std::string& fileName, size_t numCassesI):
numClasses(numCassesI)
{
	dimensions = 5+numClasses;
	std::cout<<"dimensions "<<dimensions;
	net = cv::dnn::readNet(fileName);
	net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
	net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

std::vector<Yolo5::DetectedClass> Yolo5::detect(const cv::Mat& image)
{
	cv::Mat blob;
	cv::dnn::blobFromImage(image, blob, 1.0/255, cv::Size(TRAIN_SIZE_X, TRAIN_SIZE_Y), cv::Scalar(), true, true);

	printMatInfo(blob, "blob");

	net.setInput(blob);

	std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

	double xScale = image.cols/static_cast<double>(TRAIN_SIZE_X);
	double yScale = image.rows/static_cast<double>(TRAIN_SIZE_X);

	float* dataPtr = (float *)outputs[0].data;

	std::vector<int> classNums;
	std::vector<float> probs;
	std::vector<cv::Rect> boxes;

	std::vector<DetectedClass> detections;

	for (int i = 0; i < YOLO_N_VECTOR_DEPTH; ++i)
	{
		float prob = dataPtr[4];
		if(prob > DETECTION_THRESH)
		{
			float* scoresPtr = dataPtr + 5;
			cv::Mat scores(1, numClasses, CV_32FC1, scoresPtr);
			cv::Point classId;
			double maxClassScore;
			cv::minMaxLoc(scores, 0, &maxClassScore, 0, &classId);
			if (maxClassScore > DETECTION_THRESH)
			{

				probs.push_back(prob);
				classNums.push_back(classId.x);

				float x = dataPtr[0];
				float y = dataPtr[1];
				float w = dataPtr[2];
				float h = dataPtr[3];
				int left = (x - 0.5 * w) * xScale;
				int top = (y - 0.5 * h) * yScale;
				int width = w * xScale;
				int height = h * yScale;
				boxes.push_back(cv::Rect(left, top, width, height));
			}

		}
		dataPtr += dimensions;
	}

	std::vector<int> indices;
	cv::dnn::NMSBoxes(boxes, probs, SCORE_THRES, NMS_THRESH, indices);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int index = indices[i];
		DetectedClass result;
		result.classId = classNums[index];
		result.prob = probs[index];
		result.rect = boxes[index];
		detections.push_back(result);
	}
	return detections;
}

void Yolo5::drawDetection(cv::Mat& image, const DetectedClass& detection)
{
	cv::rectangle(image, detection.rect, cv::Scalar(detection.prob*255,0,255), 2);
	cv::putText(image, std::to_string(detection.classId) + ": " + std::to_string(detection.prob),
		cv::Point(detection.rect.x, detection. rect.y-3), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255,0,0), 1, cv::LINE_8, false);
}
