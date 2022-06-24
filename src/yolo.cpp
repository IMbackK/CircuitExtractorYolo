#include "yolo.h"
#include <vector>

Yolo5::Yolo5(const cv::dnn::Net &netI):
net(netI)
{
	net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
	net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

Yolo5::Yolo5(const std::string& fileName)
{
	net = cv::dnn::readNet(fileName);
}

std::vector<Yolo5::DetectedClass> Yolo5::detect(const cv::Mat& image)
{
	cv::Mat blob;
	cv::dnn::blobFromImage(image, blob, 1.0/255, cv::Size(TRAIN_SIZE_X, TRAIN_SIZE_Y), cv::Scalar(), true, false);
	net.setInput(blob);

	std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

	double xScale = image.cols/TRAIN_SIZE_X;
	double yScale = image.rows/TRAIN_SIZE_Y;

	float* dataPtr = (float *)outputs[0].data;

	const int dimensions = 85;
	const int rows = 25200;

	std::vector<int> classNums;
	std::vector<float> probs;
	std::vector<cv::Rect> boxes;

	std::vector<DetectedClass> detections;

	for (int i = 0; i < rows; ++i)
	{
		float prob = dataPtr[4];
		if(prob > DETECTION_THRESH)
		{
			float* scoresPtr = dataPtr + 5;
			cv::Mat scores(1, className.size(), CV_32FC1, scoresPtr);
			cv::Point classId;
			double maxClassScore;
			cv::minMaxLoc(scores, 0, &maxClassScore, 0, &classId);
			if (maxClassScore > DETECTION_THRESH)
			{

				DetectedClass detection;
				detection.prob = prob;
				detection.classId = classId.x;

				float x = dataPtr[0];
				float y = dataPtr[1];
				float w = dataPtr[2];
				float h = dataPtr[3];
				int left = int((x - 0.5 * w) * xScale);
				int top = int((y - 0.5 * h) * yScale);
				int width = int(w * xScale);
				int height = int(h * yScale);
				detection.rect = cv::Rect(left, top, width, height);
				detections.push_back(detection);
			}

		}
		dataPtr += dimensions;
	}

	return detections;
}


