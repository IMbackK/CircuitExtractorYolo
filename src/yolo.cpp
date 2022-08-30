#include "yolo.h"
#include <vector>
#include <string>
#include <opencv2/imgproc.hpp>
#include <assert.h>
#include <opencv2/highgui.hpp>

#include "log.h"
#include "utils.h"

Yolo5::Yolo5(const cv::dnn::Net &netI, size_t numCassesI, int trainSizeXIn, int trainSizeYIn):
numClasses(numCassesI), net(netI), trainSizeX(trainSizeXIn), trainSizeY(trainSizeYIn)
{
	dimensions = 5+numClasses;
	net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
	net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

Yolo5::Yolo5(const std::string& fileName, size_t numCassesI, int trainSizeXIn, int trainSizeYIn):
numClasses(numCassesI), trainSizeX(trainSizeXIn), trainSizeY(trainSizeYIn)
{
	dimensions = 5+numClasses;
	net = cv::dnn::readNet(fileName);
	net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
	net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

cv::Mat Yolo5::resizeWithBorder(const cv::Mat& mat)
{
	assert(mat.dims == 2);

	cv::Mat resized;
	double aspectRatio = mat.cols/static_cast<double>(mat.rows);
	cv::Mat out(trainSizeX, trainSizeY, mat.type());

	if(mat.rows < mat.cols)
	{
		cv::resize(mat, resized, cv::Size(trainSizeX, trainSizeX/aspectRatio), 0, 0, cv::INTER_LINEAR);
		int borderPix = (trainSizeX-trainSizeX/aspectRatio)/2;
		cv::copyMakeBorder(resized, out, borderPix, borderPix, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
	}
	else
	{
		cv::resize(mat, resized, cv::Size(trainSizeY*aspectRatio, trainSizeY), 0, 0, cv::INTER_LINEAR);
		int borderPix = (trainSizeY-trainSizeY*aspectRatio)/2;
		cv::copyMakeBorder(resized, out, 0, 0, borderPix, borderPix, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
	}

	cv::resize(out, out, cv::Size(trainSizeX, trainSizeY), 0, 0, cv::INTER_LINEAR);
	return out;
}

cv::Mat Yolo5::prepare(const cv::Mat& mat)
{
	cv::Mat inter = resizeWithBorder(mat);

	if(inter.channels() == 1)
		cv::cvtColor(inter, inter,  cv::COLOR_GRAY2BGR);

	if(inter.type() != CV_32F)
		inter.convertTo(inter, CV_32F, 1.0/255);

	const int dims[] = {1, inter.channels(), inter.rows, inter.cols};
	cv::Mat out(sizeof(dims)/sizeof(*dims), dims, CV_32F);


	std::vector<cv::Mat> splitChannels(3);
	cv::split(inter, splitChannels);

	for(int channel = 0; channel < inter.channels(); ++channel)
	{
		for(int row = 0; row < inter.rows; ++row)
		{
			for(int col = 0; col < inter.cols; ++col)
			{
				int outIndex = channel*inter.rows*inter.cols+row*inter.cols+col;
				assert(outIndex < static_cast<int>(out.total()));
				reinterpret_cast<float*>(out.data)[outIndex] = splitChannels[channel].at<float>(row, col);
			}
		}
	}
	return out;
}

void Yolo5::transformCord(std::vector<DetectedClass>& detections, const cv::Size& matSize)
{
	double aspectRatio = matSize.width/static_cast<double>(matSize.height);

	if(aspectRatio > 1)
	{
		double scaleFactor = static_cast<double>(matSize.width)/trainSizeX;
		int borderPix = (trainSizeY-matSize.height/scaleFactor)/2;

		for(DetectedClass& detection : detections)
		{
			detection.rect.width = detection.rect.width*scaleFactor;
			detection.rect.height = detection.rect.height*scaleFactor;

			detection.rect.x = detection.rect.x*scaleFactor;
			detection.rect.y = (detection.rect.y - borderPix)*scaleFactor;
		}
	}
	else
	{
		double scaleFactor = static_cast<double>(matSize.height)/trainSizeX;
		int borderPix = (trainSizeX-matSize.width/scaleFactor)/2;

		for(DetectedClass& detection : detections)
		{
			detection.rect.width = detection.rect.width*scaleFactor;
			detection.rect.height = detection.rect.height*scaleFactor;

			detection.rect.x = (detection.rect.x-borderPix)*scaleFactor;
			detection.rect.y = detection.rect.y*scaleFactor;
		}
	}
}

std::vector<Yolo5::DetectedClass> Yolo5::detect(const cv::Mat& image)
{
	cv::Mat blob = prepare(image);

	net.setInput(blob);

	std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

	for(size_t i = 0; i < outputs.size(); ++i)
		outputs[i] = getMatPlane(outputs[i], 0);

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
				int left = (x - 0.5 * w);
				int top = (y - 0.5 * h);
				int width = w;
				int height = h;
				boxes.push_back(cv::Rect(left, top, width, height));
			}
		}
		dataPtr += dimensions;
	}

	Log(Log::SUPERDEBUG, false)<<"boxes count "<<boxes.size();

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

	transformCord(detections, cv::Size(image.cols, image.rows));

	Log(Log::SUPERDEBUG)<<" detections count "<<detections.size();
	return detections;
}

void Yolo5::drawDetection(cv::Mat& image, const DetectedClass& detection)
{
	cv::rectangle(image, detection.rect, cv::Scalar(detection.prob*255,0,255), 2);
	cv::putText(image, std::to_string(detection.classId) + ": " + std::to_string(detection.prob),
		cv::Point(detection.rect.x, detection. rect.y-3), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255,0,0), 1, cv::LINE_8, false);
}
