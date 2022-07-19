#include "yolo.h"
#include <vector>
#include <string>
#include <opencv2/imgproc.hpp>
#include <assert.h>
#include <opencv2/highgui.hpp>

#include "log.h"
#include "utils.h"

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
	net = cv::dnn::readNet(fileName);
	net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
	net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

cv::Mat Yolo5::resizeWithBorder(const cv::Mat& mat)
{
	assert(mat.dims == 2);

	cv::Mat resized;
	double aspectRatio = mat.cols/static_cast<double>(mat.rows);
	cv::Mat out(TRAIN_SIZE_X, TRAIN_SIZE_Y, mat.type());

	if(mat.rows < mat.cols)
	{
		cv::resize(mat, resized, cv::Size(TRAIN_SIZE_X, TRAIN_SIZE_X/aspectRatio), 0, 0, cv::INTER_LINEAR);
		int borderPix = (TRAIN_SIZE_X-TRAIN_SIZE_X/2)/2;
		Log(Log::SUPERDEBUG)<<"mat.rows "<<mat.rows<<" mat.cols "<<mat.cols<<" borderPix "<<borderPix;
		cv::copyMakeBorder(resized, out, borderPix, borderPix, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
	}
	else
	{
		cv::resize(mat, resized, cv::Size(TRAIN_SIZE_Y*aspectRatio, TRAIN_SIZE_Y), 0, 0, cv::INTER_LINEAR);
		int borderPix = (TRAIN_SIZE_Y-TRAIN_SIZE_Y*aspectRatio)/2;
		Log(Log::SUPERDEBUG)<<"mat.rows "<<mat.rows<<" mat.cols "<<mat.cols<<" borderPix "<<borderPix;
		cv::copyMakeBorder(resized, out, 0, 0, borderPix, borderPix, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
	}

	cv::resize(out, out, cv::Size(TRAIN_SIZE_X, TRAIN_SIZE_Y), 0, 0, cv::INTER_LINEAR);
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

	printMatInfo(out, __func__ + std::string(" out:"));

	std::vector<cv::Mat> splitChannels(3);
	cv::split(inter, splitChannels);

	for(size_t i = 0; i < splitChannels.size(); ++i)
	{
		printMatInfo(splitChannels[i], __func__ + std::string(" split ") + std::to_string(i) + ":");
	}

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

std::vector<Yolo5::DetectedClass> Yolo5::detect(const cv::Mat& image)
{
	//cv::Mat blob;
	//cv::dnn::blobFromImage(image, blob, 1.0/255, cv::Size(TRAIN_SIZE_X, TRAIN_SIZE_Y), cv::Scalar(0, 0, 0), false, false);

	cv::Mat blob = prepare(image);

	std::cout<<"STEP: "<<blob.step1()<<std::endl;

	cv::Mat mat3d = getMatPlane4d(blob, 0);
	cv::Mat mat2d = getMatPlane(mat3d, 0);
	//printMat(mat2d, Log::INFO);

	if(Log::level == Log::SUPERDEBUG)
	{
		Log(Log::SUPERDEBUG)<<"Mat2d";
		cv::Mat viz;
		mat2d.convertTo(viz, CV_8U, 255);
		cv::imshow("Viewer", viz);
		cv::waitKey(0);
	}

	net.setInput(blob);

	std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

	double xScale = image.cols/static_cast<double>(TRAIN_SIZE_X);
	double yScale = image.rows/static_cast<double>(TRAIN_SIZE_Y);

	for(size_t i = 0; i < outputs.size(); ++i)
	{
		outputs[i] = getMatPlane(outputs[i], 0);
		Log(Log::INFO, false)<<"output "<<i<<' ';
		//printMat(outputs[i], Log::INFO);
	}

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

	Log(Log::SUPERDEBUG)<<" detections count "<<detections.size();
	return detections;
}

void Yolo5::drawDetection(cv::Mat& image, const DetectedClass& detection)
{
	cv::rectangle(image, detection.rect, cv::Scalar(detection.prob*255,0,255), 2);
	cv::putText(image, std::to_string(detection.classId) + ": " + std::to_string(detection.prob),
		cv::Point(detection.rect.x, detection. rect.y-3), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255,0,0), 1, cv::LINE_8, false);
}
