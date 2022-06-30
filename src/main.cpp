#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <poppler-document.h>
#include <poppler-image.h>
#include <filesystem>
#include <vector>
#include <future>

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

			if(Log::level == Log::SUPERDEBUG)
			{
				cv::imshow("Viewer", circuts.back()-1);
				cv::waitKey(0);
				Yolo5::drawDetection(visulization, detection);
			}
		}

		if(detections.size() > 0 && Log::level == Log::SUPERDEBUG)
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
	E_TYPE_UNKOWN,
	E_TYPE_COUNT,
} ElementType;

struct Element
{
	ElementType type;
	cv::Rect rect;
	cv::Mat image;
};

class Circut
{
public:

	std::string model;
	float prob;
	cv::Mat image;
	std::vector<Element> elements;

	cv::Mat ciructImage() const
	{
		cv::Mat visulization;
		image.copyTo(visulization);
		for(size_t i = 0; i < elements.size(); ++i)
		{
			cv::rectangle(image, elements[i].rect, cv::Scalar(0,0,255), 2);
			cv::putText(image, std::to_string(static_cast<int>(elements[i].type)),
				cv::Point(elements[i].rect.x, elements[i].rect.y-3),
				cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255,0,0), 1, cv::LINE_8, false);
		}
		return visulization;
	}
};

class Document
{
public:
	std::string title;
	std::string keywords;
	std::string basename;
	std::string author;
	std::vector<Circut> circuts;

	void print(Log::Level level) const
	{
		Log(level)<<"Document: "<<basename<<" \""<<title<<"\" by \""<<author<<'\"';
		if(keywords.size() > 0)
		Log(level)<<"keywords: "<<keywords;
	}

	bool saveCircutImages(const std::filesystem::path folder) const
	{
		if(!std::filesystem::is_directory(folder))
		{
			if(!std::filesystem::create_directory(folder))
			{
				Log(Log::ERROR)<<folder<<" is not a valid directory and no directory could be created at this location";
				return false;
			}
		}
		for(size_t i = 0; i < circuts.size(); ++i)
		{
			const Circut& circut = circuts[i];
			std::filesystem::path path = folder /
				std::filesystem::path(basename + "_" +
				std::to_string(i) + "P" +
				std::to_string(circut.prob) + ".png");
			try
			{
				cv::imwrite(path, circut.ciructImage());
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

void getCircutElements(Circut& circut, Yolo5* yolo)
{
	std::vector<Yolo5::DetectedClass> detections = yolo->detect(circut.image);
	Log(Log::DEBUG)<<"Elements: "<<detections.size();
	for(const Yolo5::DetectedClass& detection : detections)
	{
		Element element;
		element.image = circut.image(detection.rect);
		element.type = static_cast<ElementType>(detection.classId);
		element.rect = detection.rect;
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
		if(circuts[i].elements.size() < 2)
		{
			circuts.erase(circuts.begin()+i);
			--i;
		}
	}
}

void cleanDocuments(std::vector<Document>& documents)
{
	for(size_t i = 0; i < documents.size(); ++i)
	{
		cleanCircuts(documents[i].circuts);
		if(documents[i].circuts.empty())
		{
			documents.erase(documents.begin()+i);
			--i;
		}
	}
}

std::string getNextString(int argc, char** argv, int& argvCounter)
{
	while(argvCounter < argc)
	{
		std::string str(argv[argvCounter]);
		++argvCounter;
		if(str != "-v")
			return str;
		else
			Log::level = Log::DEBUG;
	}
	return "";
}

bool processDoucment(const std::string& documentFileName, Yolo5 circutYolo, Yolo5 elementYolo)
{
	Document document = getCircutDocument(documentFileName, &circutYolo);
	for(Circut& circut : document.circuts)
	{
		getCircutElements(circut, &circutYolo);
	}
	bool ret = document.saveCircutImages("./circuts");
	if(!ret)
		Log(Log::WARN)<<"Error saving files for "<<document.basename;
	return ret;
}

int main(int argc, char** argv)
{
	Log::level = Log::INFO;
	if(argc < 4)
	{
		printUsage(argc, argv);
		return 1;
	}

	Yolo5* circutYolo;
	Yolo5* elementYolo;

	int argvCounter = 1;
	std::string circutNetworkFileName = getNextString(argc, argv, argvCounter);
	if(circutNetworkFileName.empty())
	{
		printUsage(argc, argv);
		return 1;
	}
	try
	{
		circutYolo = new Yolo5(circutNetworkFileName, 1);
		Log(Log::INFO)<<"Red circut network from "<<circutNetworkFileName;
	}
	catch(const cv::Exception& ex)
	{
		Log(Log::ERROR)<<"Can not read network from "<<circutNetworkFileName;
		return 1;
	}

	std::string elementNetworkFileName = getNextString(argc, argv, argvCounter);
	if(elementNetworkFileName.empty())
	{
		printUsage(argc, argv);
		return 1;
	}
	try
	{
		elementYolo = new Yolo5(elementNetworkFileName, 1);
		Log(Log::INFO)<<"Red element network from "<<elementNetworkFileName;
	}
	catch(const cv::Exception& ex)
	{
		Log(Log::ERROR)<<"Can not read network from "<<elementNetworkFileName;
		return 1;
	}

	std::vector<std::string> fileNames;
	fileNames.reserve(argc - argvCounter);
	for(int i = argvCounter; i < argc; ++i)
	{
		if(std::string(argv[i]) == "-v")
			Log::level = Log::DEBUG;
		fileNames.push_back(argv[i]);
	}

	if(Log::level == Log::SUPERDEBUG)
	{
		cv::namedWindow( "Viewer", cv::WINDOW_NORMAL );
		cv::resizeWindow("Viewer", 960, 500);
	}

	std::vector<std::shared_future<bool>> futures;
	futures.reserve(64);
/*
	for(size_t i = 0; i < fileNames.size(); ++i)
	{
		Log(Log::INFO)<<"Starting work on document "<<i<<" of "<<fileNames.size();
		processDoucment(fileNames[i], *circutYolo, *elementYolo);
		Log(Log::INFO)<<"Finished document";
	}
*/
	for(size_t i = 0; i < fileNames.size(); ++i)
	{
		while(i < fileNames.size() && futures.size() < 64)
		{
			futures.push_back(std::async(std::launch::async, processDoucment, fileNames[i], *circutYolo, *elementYolo));
			Log(Log::INFO)<<"Starting work on document "<<i<<" of "<<fileNames.size();
			++i;
		}

		for(size_t j = 0; j < futures.size(); ++j)
		{
			if(futures[j].wait_for(std::chrono::microseconds(0)) == std::future_status::ready)
			{
				futures.erase(futures.begin()+j);
				Log(Log::INFO)<<"Finished document";
				break;
			}
		}
	}

	for(size_t j = 0; j < futures.size(); ++j)
	{
		futures[j].wait();
		Log(Log::INFO)<<"Finished document";
	}



	delete circutYolo;
	delete elementYolo;

	return 0;
}
