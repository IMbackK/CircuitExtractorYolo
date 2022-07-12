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

typedef enum
{
	ALGO_INVALID = -1,
	ALGO_CIRCUT = 0,
	ALGO_ELEMENT,
	ALGO_LINE,
	ALGO_COUNT
} Algo;

void printUsage(int argc, char** argv)
{
	Log(Log::INFO)<<"Usage: "<<argv[0]<<"[ALGO] [IMAGEFILENAME]";
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
	}
	catch(const std::invalid_argument& ex)
	{
		if(in == "circut")
			out = ALGO_CIRCUT;
		else if(in == "element")
			out = ALGO_ELEMENT;
		else if(in == "line")
			out = ALGO_LINE;
		else
			out = ALGO_INVALID;
	}
	return out;
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

	if(Log::level == Log::SUPERDEBUG)
		cv::namedWindow( "Viewer", cv::WINDOW_NORMAL );

	cv::Mat image = cv::imread(argv[2]);
	if(!image.data)
	{
		Log(Log::ERROR)<<argv[2]<<" is not a valid image file";
		return 2;
	}

	switch(algo)
	{
		case ALGO_CIRCUT:
			break;
		case ALGO_ELEMENT:
			break;
		case ALGO_LINE:
			lineDetect(image);
			break;
		case ALGO_INVALID:
		default:
			Log(Log::ERROR)<<'\"'<<argv[1]<<"\" is not a valid algorithm";
			return 3;
	}


	return 0;
}
