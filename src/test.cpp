#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <poppler-document.h>
#include <poppler-image.h>
#include <poppler-global.h>
#include <filesystem>
#include <vector>
#include <future>
#include <memory>
#include <string>

#include "log.h"
#include "document.h"
#include "circut.h"
#include "randomgen.h"
#include "linedetection.h"
#include "yolo.h"
#include "document.h"

static constexpr char circutNetworkFileName[]  = "../data/networks/circut/640/best.onnx";
static constexpr char elementNetworkFileName[] = "../data/networks/element/640/best.onnx";
static constexpr char graphNetworkFileName[] = "../data/networks/graph/640/best.onnx";

typedef enum
{
	ALGO_INVALID = -1,
	ALGO_CIRCUT = 0,
	ALGO_ELEMENT,
	ALGO_NET,
	ALGO_GRAPH,
	ALGO_COUNT
} Algo;

void printUsage(int argc, char** argv)
{
	Log(Log::INFO)<<"Usage: "<<argv[0]<<" [ALGO] [IMAGEFILENAME]";
	Log(Log::INFO)<<"Valid algos: circut, element, net, graph";
}

Algo parseAlgo(const std::string& in)
{
	Algo out = ALGO_INVALID;
	try
	{
		int tmp = std::stoi(in);
		if(tmp >= ALGO_COUNT || tmp < 0)
		{
			const char msg[] = "Algo enum out of range";
			Log(Log::ERROR)<<msg;
			throw std::invalid_argument(msg);
		}
		out = static_cast<Algo>(tmp);
	}
	catch(const std::invalid_argument& ex)
	{
		if(in == "circut")
			out = ALGO_CIRCUT;
		else if(in == "element")
			out = ALGO_ELEMENT;
		else if(in == "net")
			out = ALGO_NET;
		else if(in == "graph")
			out = ALGO_GRAPH;
		else
			out = ALGO_INVALID;
	}
	return out;
}

void algoCircut(cv::Mat& image)
{
	Yolo5* yolo;
	try
	{
		yolo = new Yolo5(circutNetworkFileName, 1, 640, 640);
	}
	catch(const cv::Exception& ex)
	{
		Log(Log::ERROR)<<"Can not read network from "<<circutNetworkFileName;
		return;
	}

	std::vector<cv::Mat> images({image});
	std::vector<cv::Mat> detections = getYoloImages(images, yolo);

	delete yolo;
}

void algoElement(cv::Mat& image)
{
	Yolo5* yolo;
	try
	{
		yolo = new Yolo5(elementNetworkFileName, 7, 640, 640);
	}
	catch(const cv::Exception& ex)
	{
		Log(Log::ERROR)<<"Can not read network from "<<elementNetworkFileName;
		return;
	}

	Circut circut;
	circut.image = extendBorder(image, 15);

	if(Log::level == Log::SUPERDEBUG)
	{
		cv::imshow("Viewer", circut.image);
		cv::waitKey(0);
	}

	circut.detectElements(yolo);

	if(Log::level == Log::SUPERDEBUG)
	{
		cv::imshow("Viewer", circut.ciructImage());
		cv::waitKey(0);
	}

	delete yolo;
}

void algoLine(cv::Mat& image)
{
	Yolo5* yolo;
	try
	{
		yolo = new Yolo5(elementNetworkFileName, 7, 640, 640);
	}
	catch(const cv::Exception& ex)
	{
		Log(Log::ERROR)<<"Can not read network from "<<elementNetworkFileName;
		return;
	}

	Circut circut;
	circut.image = extendBorder(image, 15);

	circut.detectElements(yolo);

	circut.detectNets();

	circut.setDirectionHint(circut.estimateDirection());

	circut.parseCircut();

	std::string modelString = circut.getString();

	if(Log::level == Log::SUPERDEBUG)
	{
		cv::imshow("Viewer", circut.ciructImage());
		cv::waitKey(0);
	}

	Log(Log::INFO)<<"Parsed string: "<<modelString;
}

void algoGraph(cv::Mat& image)
{
	Yolo5* yolo;
	try
	{
		yolo = new Yolo5(graphNetworkFileName, 1, 640, 640);
	}
	catch(const cv::Exception& ex)
	{
		Log(Log::ERROR)<<"Can not read network from "<<graphNetworkFileName;
		return;
	}

	std::vector<cv::Mat> images({image});
	std::vector<cv::Mat> detections = getYoloImages(images, yolo);

	delete yolo;
}

int main(int argc, char** argv)
{
	rd::init();
	Log::level = Log::SUPERDEBUG;
	if(argc != 3)
	{
		printUsage(argc, argv);
		return 1;
	}

	Algo algo = parseAlgo(argv[1]);

	cv::Mat image = cv::s(argv[2]);
	if(!image.data)
	{
		Log(Log::ERROR)<<argv[2]<<" is not a valid image file";
		return 2;
	}

	if(Log::level == Log::SUPERDEBUG)
	{
		cv::namedWindow( "Viewer", cv::WINDOW_NORMAL );
		cv::imshow("Viewer", image);
		cv::waitKey(0);
	}

	switch(algo)
	{
		case ALGO_CIRCUT:
			algoCircut(image);
			break;
		case ALGO_ELEMENT:
			algoElement(image);
			break;
		case ALGO_NET:
			algoLine(image);
			break;
		case ALGO_GRAPH:
			algoGraph(image);
			break;
		case ALGO_INVALID:
		default:
			Log(Log::ERROR)<<'\"'<<argv[1]<<"\" is not a valid algorithm";
			return 3;
	}

	return 0;
}
