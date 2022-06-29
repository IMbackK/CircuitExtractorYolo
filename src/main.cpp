#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <poppler-document.h>
#include <poppler-image.h>
#include <filesystem>

#include "log.h"
#include "popplertocv.h"
#include "yolo.h"

void printUsage(int argc, char** argv)
{
	Log(Log::INFO)<<"Usage: "<<argv[0]<<"[CIRCUTNETWORKFILENAME] [ELEMENTNETWOKRFILENAME] [PDFFILENAME]";
}

std::vector<cv::Mat> getCircutImages(std::vector<cv::Mat> images, Yolo5* yolo, std::vector<float>* probs)
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
			try
			{
				circuts.push_back(cv::Mat(image, detection.rect));
				if(probs)
					probs->push_back(detection.prob);
			}
			catch(const cv::Exception& ex)
			{
				Log(Log::WARN)<<"Failed to process rect "<<ex.what();
			}

			if(Log::level == Log::DEBUG)
			{
				cv::imshow("Viewer", circuts.back()-1);
				cv::waitKey(0);
				Yolo5::drawDetection(visulization, detection);
			}
		}

		if(detections.size() > 0 && Log::level == Log::DEBUG)
		{
			cv::imshow("Viewer", visulization);
			cv::waitKey(0);
		}
	}

	return circuts;
}

typedef enum {
	E_TYPE_R = 0,
	E_TYPE_C,
	E_TYPE_L,
	E_TYPE_P,
	E_TYPE_W,
	E_TYPE_SOURCE,
	E_TYPE_UNKOWN
	E_TYPE_COUNT,
} ElementType;

struct Element
{
	ElementType type;
	cv::Rect rect;
	cv::Mat image;
}

struct Circut
{
	std::string model;
	float prob;
	cv::Mat image;
	std::vector<Element> elements;
};

class Document
{
public:
	std::string title;
	std::string keywords;
	std::string basename;
	std::string author;
	std::vector<Circut> circuts;

	void print(Log::Level level)
	{
		Log(level)<<"Document: "<<basename<<" \""<<title<<"\" by \""<<author<<'\"';
		if(keywords.size() > 0)
		Log(level)<<"keywords: "<<keywords;
	}
};

Document getCircutDocument(const std::string& fileName, Yolo5* yolo)
{
	poppler::document* popdocument = poppler::document::load_from_file(fileName);

	Document document;

	if(!popdocument)
	{
		Log(Log::ERROR)<<"Could not load pdf file from "<<fileName;
		return document;
	}

	if(popdocument->is_encrypted())
	{
		Log(Log::ERROR)<<"Only unencrypted files are supported";
		return document;
	}

	document.keywords = popdocument->get_keywords().to_latin1();
	document.title = popdocument->get_title().to_latin1();
	document.author = popdocument->get_creator().to_latin1();
	document.basename = std::filesystem::path(fileName).filename();
	document.print(Log::INFO);

	std::vector<cv::Mat> pageImages = getMatsFromDocument(popdocument, cv::Size(1280, 1280));
	std::vector<float> probs;
	std::vector<cv::Mat> circutImages = getCircutImages(pageImages, yolo, &probs);

	for(size_t i = 0; i < circutImages.size(); ++i)
	{
		const cv::Mat& image = circutImages[i];
		Circut circut;
		circut.image = image;
		circut.prob = probs[i];
		document.circuts.push_back(circut);
	}
	return document;
}

bool saveDocumentImages(const std::filesystem::path folder, const Document& document)
{
	if(!std::filesystem::is_directory(folder))
	{
		if(!std::filesystem::create_directory(folder))
		{
			Log(Log::ERROR)<<folder<<" is not a valid directory and no directory could be created at this location";
			return false;
		}
	}
	for(size_t i = 0; i < document.circuts.size(); ++i)
	{
		const Circut& circut = document.circuts[i];
		std::filesystem::path path = folder / std::filesystem::path(document.basename + "_" + std::to_string(i) + "P" + std::to_string(circut.prob) + ".png");
		try
		{
			cv::imwrite(path, circut.image);
		}
		catch(const cv::Exception& ex)
		{
			Log(Log::ERROR)<<"Cant write "<<path<<' '<<ex.what();
			return false;
		}
		Log(Log::INFO)<<"Wrote image to "<<path;
	}
	return true;
}

void getCircutElements(Circut& circut, Yolo5* yolo)
{
	std::vector<Yolo5::DetectedClass> detections = yolo.detect(circut.image);
	for(const Yolo5::DetectedClass& detection : detections)
	{
		Element element;
		element.image = circut.image(detection.rect);
		element.type = detection.classId;
		element.rect = circut.rect;
		if(Log::level == Log::DEBUG)
		{
			cv::Mat visulization;
			circut.image.copyTo(visulization);
			Yolo5::drawDetection(visulization, detection);
		}
		circut.elements.push_back(element);
	}
}

void cleanCircuts(std::vector<Circut>& circuts)
{
	for(size_t i = 0; i < circuts.size(); ++i)
	{
		if(circuts.elements.size() < 2)
		{
			circuts.erase(circuts.begin()+i);
			--i;
		}
	}
}

std::string getNextString(int argc, char** argv, int& argvCounter)
{
	while(argvCounter < argc)
	{
		std::string str(argv[argvCounter]);
		if(str != "-v")
			return str;
		else
			Log::level = Log::DEBUG;
		++argvCounter;
	}
	return "";
}

int main(int argc, char** argv)
{
	Log::level = Log::INFO;
	if(argc < 4)
	{
		printUsage(argc, argv);
		return 1;
	}

	int argvCounter = 1;
	std::string circutNetworkFileName = getNextString(argc, argv, argvCounter);
	if(circutNetworkFileName.empty())
	{
		printUsage(argc, argv);
		return 1;
	}
	try
	{
		Yolo5 circutYolo(circutNetworkFileName, 1);
	}
	catch(const cv::Exception& ex)
	{
		Log(Log::ERROR)<<"Can not read network from "<<circutNetworkFileName;
		return 1
	}

	std::string elementNetworkFileName = getNextString(argc, argv, argvCounter);
	if(elementNetworkFileName.empty())
	{
		printUsage(argc, argv);
		return 1;
	}
	try
	{
		Yolo5 elementYolo(elementNetworkFileName, E_TYPE_COUNT);
	}
	catch(const cv::Exception& ex)
	{
		Log(Log::ERROR)<<"Can not read network from "<<elementNetworkFileName;
		return 1
	}

	std::vector<std::string> fileNames;
	fileNames.reserve(argc - 1);
	for(int i = 2; i < argc; ++i)
	{
		if(std::string(argv[i]) == "-v")
			Log::level = Log::DEBUG;
		fileNames.push_back(argv[i]);
	}

	if(Log::level == Log::DEBUG)
	{
		cv::namedWindow( "Viewer", cv::WINDOW_NORMAL );
		cv::resizeWindow("Viewer", 960, 500);
	}

	for(size_t i = 0; i < fileNames.size(); ++i)
	{
		Log(Log::INFO)<<"Processing document "<<i<<" of "<<fileNames.size();
		Document document = getCircutDocument(fileNames[i], &yolo);
		bool ret = saveDocumentImages("./", document);
		if(!ret)
			Log(Log::WARN)<<"Error saving files for "<<document.basename;
	}

	return 0;
}
