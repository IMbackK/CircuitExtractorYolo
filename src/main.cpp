#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <poppler-document.h>
#include <poppler-image.h>

#include "log.h"
#include "popplertocv.h"
#include "yolo.h"

void printUsage(int argc, char** argv)
{
	Log(Log::INFO)<<"Usage: "<<argv[0]<<" [PDFFILENAME] [NETWORKFILENAME]";
}

std::vector<cv::Mat> getCircuts(std::vector<cv::Mat> images, Yolo5* yolo)
{
	std::vector<cv::Mat> circuts;

	for(cv::Mat& image : images)
	{
		std::vector<Yolo5::DetectedClass> detections = yolo->detect(image);
		cv::Mat visulization;

		if(Log::level == Log::DEBUG)
			image.copyTo(visulization);

		for(const Yolo5::DetectedClass& detection : detections)
		{
			circuts.push_back(cv::Mat(image, detection.rect));

			if(Log::level == Log::DEBUG)
			{
				cv::imshow("Viewer", circuts.back()-1);
				cv::waitKey(0);
				Yolo5::drawDetection(visulization, detection);
			}
		}

		if(Log::level == Log::DEBUG)
		{
			cv::imshow("Viewer", visulization);
			cv::waitKey(0);
		}
	}

	return circuts;
}

std::vector<cv::Mat> getCircutsFromDocuments(const std::vector<std::string>& fileNames, Yolo5* yolo)
{
	std::vector<cv::Mat> output;
	for(const std::string& fileName : fileNames)
	{
		poppler::document* document = poppler::document::load_from_file(fileName);

		if(!document)
		{
			Log(Log::ERROR)<<"Could not load pdf file from "<<fileName;
			continue;
		}

		if(document->is_encrypted())
		{
			Log(Log::ERROR)<<"Only unencrypted files are supported";
			continue;
		}

		std::string keywords = document->get_keywords().to_latin1();
		Log(Log::INFO)<<"Got PDF with "<<document->pages()<<" pages";
		if(!keywords.empty())
			Log(Log::INFO)<<"With keywords: "<<keywords;

		std::vector<cv::Mat> images = getMatsFromDocument(document, cv::Size(1280, 1280));
		std::vector<cv::Mat> circuts = getCircuts(images, yolo);
		output.insert(output.end(), circuts.begin(), circuts.end());

		delete document;
	}
	return output;
}

int main(int argc, char** argv)
{
	Log::level = Log::DEBUG;
	if(argc < 3)
	{
		Log(Log::ERROR)<<"A pdf file name and network file name must be provided";
		return 1;
	}

	Yolo5 yolo(argv[1], 1);

	cv::namedWindow( "Viewer", cv::WINDOW_NORMAL );
	cv::resizeWindow("Viewer", 960, 500);

	for(int i = 2; i < argc; ++i)
	{
		poppler::document* document = poppler::document::load_from_file(argv[1]);

		if(!document)
		{
			Log(Log::ERROR)<<"Could not load pdf file from "<<argv[1];
			continue;
		}

		if(document->is_encrypted())
		{
			Log(Log::ERROR)<<"Only unencrypted files are supported";
			continue;
		}

		std::string keywords = document->get_keywords().to_latin1();
		Log(Log::INFO)<<"Got PDF with "<<document->pages()<<" pages";
		if(!keywords.empty())
			Log(Log::INFO)<<"With keywords: "<<keywords;

		std::vector<cv::Mat> images = getMatsFromDocument(document, cv::Size(1280, 1280));

		delete document;
	}

	return 0;
}
